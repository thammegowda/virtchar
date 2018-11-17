# Tensor 2 Tensor aka Attention is all you need
# Thanks to http://nlp.seas.harvard.edu/2018/04/03/attention.html
import copy
import math
import time
from typing import Callable, Optional, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from virtchar import device, log, my_tensor as tensor, DialogExperiment
from virtchar.model import DialogModel
from virtchar.model.trainer import TrainerState, SteppedTrainer
from virtchar.tool.dataprep import DialogMiniBatch
from virtchar.tool.dataprep import PAD_TOK_IDX as pad_value



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model: int, vocab: int):
        super(Generator, self).__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda _x: self.self_attn(_x, _x, _x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask))
        x = self.sublayer[1](x, lambda _x: self.src_attn(_x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer: DecoderLayer, N: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class T2TModel(DialogModel):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed, tgt_embed,
                 generator: Generator):
        super(T2TModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.tgt_vocab = generator.vocab

    @property
    def model_dim(self):
        return self.generator.d_model

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        enc_outs = self.encode(src, src_mask)
        return self.decode(enc_outs, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    @staticmethod
    def make_model(src_vocab, tgt_vocab, n_layers=6, hid_size=512, ff_size=2048, n_heads=8,
                   dropout=0.1, tied_emb='three-way'):
        "Helper: Construct a model from hyper parameters."

        # args for reconstruction of model
        args = {'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'n_layers': n_layers,
                'hid_size': hid_size,
                'ff_size': ff_size,
                'n_heads': n_heads,
                'dropout': dropout,
                'tied_emb': tied_emb
                }
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_heads, hid_size)
        ff = PositionwiseFeedForward(hid_size, ff_size, dropout)

        encoder = Encoder(EncoderLayer(hid_size, c(attn), c(ff), dropout), n_layers)
        decoder = Decoder(DecoderLayer(hid_size, c(attn), c(attn), c(ff), dropout), n_layers)

        src_emb = nn.Sequential(Embeddings(hid_size, src_vocab), PositionalEncoding(hid_size, dropout))
        tgt_emb = nn.Sequential(Embeddings(hid_size, tgt_vocab), PositionalEncoding(hid_size, dropout))
        generator = Generator(hid_size, tgt_vocab)

        model = T2TModel(encoder, decoder, src_emb, tgt_emb, generator)
        if tied_emb:
            assert src_vocab == tgt_vocab
            if tied_emb == 'three-way':
                log.info("Tying the embedding weights, three ways: (SrcIn == TgtIn == TgtOut")
                model.src_embed[0].lut.weight = model.tgt_embed[0].lut.weight
                model.generator.proj.weight = model.tgt_embed[0].lut.weight
            elif tied_emb == 'two-way':
                log.info("Tying the embedding weights, two ways: (SrcIn == TgtIn")
                model.src_embed[0].lut.weight = model.tgt_embed[0].lut.weight
            else:
                raise Exception('Invalid argument to tied_emb; Known: {three-way, two-way}')

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model, args


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class LabelSmoothing(nn.Module):
    """
    Label smoothing
    """
    def __init__(self, vocab_size: int, padding_idx: int, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self._size = vocab_size
        assert 0.0 <= smoothing <= 1.0
        self.padding_idx = padding_idx
        self.criterion = nn.KLDivLoss(size_average=False)
        fill_val = smoothing / (vocab_size - 2)
        one_hot = torch.full(size=(1, vocab_size), fill_value=fill_val, device=device)
        one_hot[0][self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot)
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        # 'x' is log probabilities, originally [B, T, V], but here [B.T, V]
        # 'target' is expected word Ids, originally [B, T] but here [B.T]
        assert x.size(1) == self._size
        gtruth = target.view(-1)
        tdata = gtruth.data

        mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
        tdata_2d = tdata.unsqueeze(1)

        log_likelihood = torch.gather(x.data, 1, tdata_2d)

        smoothed_truth = self.one_hot.repeat(gtruth.size(0), 1)
        smoothed_truth.scatter_(1, tdata_2d, self.confidence)

        if mask.numel() > 0:
            log_likelihood.index_fill_(0, mask, 0)
            smoothed_truth.index_fill_(0, mask, 0)
        loss = self.criterion(x, smoothed_truth)

        # loss is a scalar value (0-dim )
        # but data parallel expects tensors (for gathering along a dim), so doing this
        return loss.unsqueeze(0)


class SimpleLossFunction:
    """
    A simple loss function that computes the loss using the criterion given
    """

    def __init__(self, generator, criterion, opt):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x_feats, y_seqs, norm, train_mode=True):

        x_probs = self.generator(x_feats)  # B x T x D --> B x T x V

        scores = x_probs.contiguous().view(-1, x_probs.size(-1))    # B x T x V --> B.T x V
        truth = y_seqs.contiguous().view(-1)  # B x T --> B.T
        assert norm != 0
        loss = self.criterion(scores, truth).sum() / norm
        if train_mode:  # dont do this for validation set
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return loss.item() * norm


class MultiGPULossFunction(SimpleLossFunction):
    """
    Loss function that uses Multiple GPUs
    Currently uses DataParallel, but this is only the early version
    """
    def __init__(self, generator, criterion, devices, opt, out_device=None):
        super(MultiGPULossFunction, self).__init__(generator, criterion, opt)
        self.multi_gpu = False
        if len(devices) > 1:
            self.multi_gpu = True
            self.device_ids = devices
            self.out_device = out_device if out_device is not None else devices[0]
            # Send out to different gpus.
            self.criterion = nn.parallel.replicate(criterion, devices=devices)
            self.generator = nn.parallel.replicate(generator, devices=self.device_ids)

    def __call__(self, outs, targets, norm, train_mode=True):
        if not self.multi_gpu:
            # let the parent class deal with this
            return super(MultiGPULossFunction, self).__call__(outs, targets, norm, train_mode)

        # FIXME: there seems to be a bug in this below code
        # TODO: generate outputs in chunks
        batch_dim = 0
        assert outs.shape[batch_dim] == targets.shape[batch_dim]
        sct_outs = nn.parallel.scatter(outs, target_gpus=self.device_ids, dim=batch_dim)
        sct_tgts = nn.parallel.scatter(targets, target_gpus=self.device_ids, dim=batch_dim)
        assert len(sct_outs) == len(sct_tgts)
        sct_generators = self.generator[:len(sct_outs)]
        sct_criteria = self.criterion[:len(sct_outs)]
        sct_preds = nn.parallel.parallel_apply(sct_generators, sct_outs)
        pairs = [(pred.contiguous().view(-1, pred.size(-1)),
                  tgt.contiguous().view(-1)) for pred, tgt in zip(sct_preds, sct_tgts)]
        sct_losses = nn.parallel.parallel_apply(sct_criteria, pairs)
        sent_losses = nn.parallel.gather(sct_losses, target_device=self.out_device, dim=batch_dim)
        total_loss = (sent_losses.sum() / norm)
        total_loss_val = total_loss.item()
        if train_mode:
            total_loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return total_loss_val * norm


class T2TTrainer(SteppedTrainer):

    def __init__(self, exp: DialogExperiment,
                 model: Optional[T2TModel] = None,
                 optim: str = 'ADAM',
                 **optim_args):
        super().__init__(exp, model, model_factory=T2TModel.make_model, optim=optim, **optim_args)

        device_ids = list(range(torch.cuda.device_count()))
        log.info(f"Going to use {torch.cuda.device_count()} GPUs ; ids:{device_ids}")

        if len(device_ids) > 1:   # Multi GPU mode
            log.warning("Multi GPU mode <<this feature is not well tested>>")
            self.model = nn.DataParallel(self.model, dim=0, device_ids=device_ids)
        generator = self.model.generator

        criterion = LabelSmoothing(vocab_size=generator.vocab,
                                   padding_idx=pad_value,
                                   smoothing=self._smoothing)

        self.loss_func = MultiGPULossFunction(generator, criterion, devices=device_ids,
                                              opt=self.opt)

    def run_valid_epoch(self, data_iter: Iterator[DialogMiniBatch]):
        """
        :param data_iter: data iterator
        :return: loss value
        """
        start = time.time()
        total_tokens = 0
        total_loss = 0.0
        with tqdm(data_iter, total=data_iter.num_batches, unit='batch') as data_bar:
            for i, batch in enumerate(data_bar):
                batch = batch.to(device)
                num_toks = batch.y_toks
                out = self.model(batch.x_seqs, batch.y_seqs, batch.x_mask, batch.y_mask)
                # skip the BOS token in  batch.y_seqs
                loss = self.loss_func(out, batch.y_seqs_nobos, num_toks, False)
                total_loss += loss
                total_tokens += num_toks
                elapsed = time.time() - start
                data_bar.set_postfix_str(
                    f'Loss:{loss / num_toks:.4f}, {int(num_toks / elapsed)}toks/s', refresh=False)
                start = time.time()
                # force free memory
                del batch

        score = total_loss / total_tokens
        return score

    def overfit_batch(self, batch, max_iters=100, stop_loss=0.01):
        """
        Try to over fit given batch (for testing purpose only, as suggested in
         https://twitter.com/karpathy/status/1013244313327681536)
        """
        tokens = 0
        loss = float('inf')
        for i in tqdm(range(max_iters)):
            num_toks = batch.y_toks
            out = self.model(batch.x_seqs, batch.y_seqs, batch.x_mask, batch.y_mask)
            # skip the BOS token in  batch.y_seqs
            loss = self.loss_func(out, batch.y_seqs_nobos, num_toks)
            tokens += num_toks
            if abs(loss) < abs(stop_loss):
                log.info(f"Stopping early at iter {i}.. Loss = {loss:.4f}")
                return i, loss
        return max_iters - 1, loss

    def train(self, steps: int, check_point: int, batch_size: int,
              check_pt_callback: Optional[Callable]=None, fine_tune=False, **args):
        log.info(f'Going to train for {steps} epochs; batch_size={batch_size}; '
                 f'check point size:{check_point}; fine_tune={fine_tune}')
        keep_models = args.get('keep_models', 4)  # keep last _ models and delete the old

        if steps <= self.start_step:
            raise Exception(f'The model was already trained to {self.start_step} steps. '
                            f'Please increase the steps or clear the existing models')
        train_data = self.exp.get_train_data(batch_size=batch_size,
                                             steps=steps - self.start_step,
                                             shuffle=True, batch_first=True, fine_tune=fine_tune)
        val_data = self.exp.get_val_data(batch_size, shuffle=False, batch_first=True)

        train_state = TrainerState(self.model, check_point=check_point)
        train_state.train_mode(True)
        with tqdm(train_data, initial=self.start_step, total=steps, unit='batch') as data_bar:
            for batch in data_bar:
                batch = batch.to(device)
                num_toks = batch.y_toks
                self.model.zero_grad()
                out = self.model(batch.x_seqs, batch.y_seqs, batch.x_mask, batch.y_mask)
                # skip the BOS token in  batch.y_seqs
                loss = self.loss_func(out, batch.y_seqs_nobos, num_toks, True)
                self.tbd.add_scalars('training', {'step_loss': loss,
                                                  'learn_rate': self.opt.curr_lr},
                                     self.opt.curr_step)

                progress_msg, is_check_pt = train_state.step(num_toks, loss)
                progress_msg += f', LR={self.opt.curr_lr:g}'

                data_bar.set_postfix_str(progress_msg, refresh=False)
                del batch   # TODO: force free memory

                if is_check_pt:
                    train_loss = train_state.reset()
                    train_state.train_mode(False)
                    self.make_check_point(val_data, train_loss, keep_models=keep_models)
                    if check_pt_callback:
                        check_pt_callback(model=self.model,
                                          step=self.opt.curr_step,
                                          train_loss=train_loss)
                    train_state.train_mode(True)


def __test_model__():

    from virtchar.tool.dummy import DummyExperiment
    vocab_size = 14
    model, _ = T2TModel.make_model(vocab_size, vocab_size,
                                   n_layers=4, hid_size=128, ff_size=256, n_heads=4)
    if False:
        for n, p in model.named_parameters():
            print(n, p.shape)

    from virtchar.use.decoder import Decoder

    exp = DummyExperiment("work", config={'model_type': 't2t'}, read_only=True,
                          vocab_size=vocab_size)

    trainer = T2TTrainer(exp=exp, model=model, warmup_steps=200)
    decr = Decoder.new(exp, trainer.model)

    assert 2 == Batch.bos_val
    src = tensor([[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [2, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]])
    src_lens = tensor([src.size(1)] * src.size(0))

    def check_pt_callback(**args):
        res = decr.greedy_decode(src, src_lens, max_len=12)
        for score, seq in res:
            log.info(f'{score:.4f} :: {seq}')

    batch_size = 50
    steps = 500
    check_point = 10
    trainer.train(steps=steps, check_point=check_point, batch_size=batch_size,
                  check_pt_callback=check_pt_callback)


if __name__ == '__main__':
    __test_model__()
