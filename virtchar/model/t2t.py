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
from virtchar.tool.dataprep import PAD_TOK_IDX as PAD_IDX, BOS_TOK_IDX as BOS_IDX


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class ComboEmbeddings(nn.Module):
    """
    Combined embeddings for characters (i.e. speakers) and words
    """
    def __init__(self, d_model, text_vocab, char_vocab, char_emb_size=None):
        super().__init__()

        self.text_emb = nn.Embedding(text_vocab, d_model)
        self.char_emb_size = char_emb_size if char_emb_size else d_model
        self.char_emb = nn.Embedding(char_vocab, self.char_emb_size)
        self.merge = nn.Linear(self.char_emb_size + d_model, d_model)
        self.d_model = d_model

    def forward(self, pair):
        text_seqs, chars = pair
        assert len(text_seqs.shape) == 2
        assert len(chars.shape) == 1
        assert text_seqs.shape[0] == chars.shape[0]
        batch_size, seq_len = text_seqs.shape
        text_embedded = self.text_emb(text_seqs) * math.sqrt(self.d_model)
        char_embedded = self.char_emb(chars) * math.sqrt(self.char_emb_size)

        text_embedded = text_embedded.view(batch_size, seq_len, self.d_model)
        char_embedded = char_embedded.view(batch_size, 1, self.char_emb_size)
        # repeat the character embedding for each time step in sequence
        char_embedded = char_embedded.expand(batch_size, seq_len, self.char_emb_size)

        # concat word embedding and character embedding along the last dimension
        embedded = torch.cat([text_embedded, char_embedded], dim=-1)

        return self.merge(embedded)


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
        # Residual connection for both the layers:
        # 1) encoder self attn 2) feed forward layer
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
        # Residual connection for all three layers :
        # 1) decoder self attn 2) dec-enc source attn, 3) feed forward layer
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, enc_outs, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask))
        x = self.sublayer[1](x, lambda _x: self.src_attn(_x, enc_outs, enc_outs, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer: DecoderLayer, n_layers: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, tgt_prev, memory, src_mask, tgt_mask):
        x = tgt_prev
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


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
        query, key, value = [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for lin, x in zip(self.linears, (query, key, value))]

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


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=800):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class HieroTransformer(DialogModel):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, utter_encoder: Encoder,
                 ctx_encoder: Encoder,
                 decoder: Decoder,
                 combo_embs: ComboEmbeddings,
                 generator: Generator,
                 dropout: float):
        super().__init__()
        self.utter_encoder = utter_encoder
        self.ctx_encoder = ctx_encoder
        self.decoder = decoder
        self.combo_embs: ComboEmbeddings = combo_embs
        self.generator = generator
        self._model_dim = generator.d_model
        # positional encoder for the chat sequence
        self.posit_enc = PositionalEncoding(self._model_dim, dropout=dropout)

    @property
    def model_dim(self):
        return self._model_dim

    @classmethod
    def mask_pad_future(cls, seqs, pad=PAD_IDX):
        """
        Create a mask to hide padding and future words.
        :param seqs:
        :param pad: pad value
        :return:
        """
        mask = (seqs != pad).unsqueeze(1)
        max_seq_len = seqs.shape[-1]
        mask = mask & cls.subsequent_mask(max_seq_len).type_as(mask.data)
        return mask

    @staticmethod
    def subsequent_mask(size):
        """
        Mask out subsequent positions. upper diagonal elements should be zero
        :param size:
        :return: mask where positions are filled with zero for subsequent positions
        """
        # upper diagonal elements are 1s, lower diagonal and the main diagonal are zeroed
        triu = torch.triu(torch.ones(size, size, dtype=torch.int8, device=device), diagonal=1)
        # invert it
        mask = triu == 0
        mask = mask.unsqueeze(0)
        return mask

    def forward(self, batch: DialogMiniBatch, add_bos=True):
        "Take in and process masked src and target sequences."

        ctx_enc_outs, ctx_mask = self.hiero_encode(batch.utters, batch.utter_lens, batch.chars,
                                                   batch.chat_ctx_idx)
        tgt_seqs = batch.resp_seqs
        if add_bos:
            # Add BOS token at the first time step
            bos_col = torch.full(size=(tgt_seqs.shape[0], 1), fill_value=BOS_IDX,
                                 device=device, dtype=torch.long)
            tgt_seqs = torch.cat([bos_col, tgt_seqs], dim=1)
        dec_feats = self.decode(ctx_enc_outs, ctx_mask, tgt_seqs, batch.resp_chars)
        if add_bos:
            # slide left;; remove the last time step
            dec_feats = dec_feats[:, :-1]
        # this can be projected to vocabulary space to produce words
        return dec_feats

    def hiero_encode(self, utter_seqs, utter_lens, chars, chat_ctx_idx):

        n_utters, max_len = utter_seqs.shape
        utter_mask = (utter_seqs != PAD_IDX)
        # :: level 1 :: prepare
        embedded = self.combo_embs((utter_seqs, chars)).view(n_utters, max_len, -1)

        utter_encoded = self.utter_encoder(embedded, utter_mask.unsqueeze(1))
        # level 1 :: post process
        # Mask out padded time steps
        utter_encoded = utter_encoded * utter_mask.unsqueeze(2).type_as(utter_encoded)

        # Sum element wise along the time dimension
        sent_reprs = utter_encoded.sum(dim=1)
        # Divide by the sqrt of length of sentences. Why? normalize the effect of unequal length
        # Google guys did it too: https://arxiv.org/pdf/1803.11175.pdf (I learned this from them)
        sent_reprs = sent_reprs.div(utter_lens.float().sqrt().unsqueeze(1))

        # :: level 2 :: Prepare
        # Now, we need to construct chat context from these
        # Inserting 0s at the 0th row, so we can pick rows based on indices, where index=0 is pad
        padded_sent_repr = torch.cat(
            [torch.zeros(1, sent_reprs.shape[1], device=device), sent_reprs], dim=0)

        # index_select works with vector, so we flatten and then restore
        chat_input = torch.index_select(padded_sent_repr, 0, chat_ctx_idx.view(-1))
        chat_input = chat_input.view(*chat_ctx_idx.shape, -1)
        chat_input = self.posit_enc(chat_input)

        chat_mask = (chat_ctx_idx != 0).unsqueeze(1)
        chat_enc_outs = self.ctx_encoder(chat_input, chat_mask)
        return chat_enc_outs, chat_mask

    def decode(self, memory, src_mask, tgt_seqs, chars):
        tgt_mask = self.mask_pad_future(tgt_seqs)
        tgt_prev_embs = self.combo_embs((tgt_seqs, chars))
        return self.decoder(tgt_prev_embs, memory, src_mask, tgt_mask)

    @staticmethod
    def make_model(text_vocab,
                   char_vocab,
                   n_layers=6,
                   hid_size=512,
                   ff_size=2048,
                   n_heads=8,
                   dropout=0.1,
                   tied_emb='three-way'):
        "Helper: Construct a model from hyper parameters."

        # args for reconstruction of model
        args = {'text_vocab': text_vocab,
                'char_vocab': char_vocab,
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
        utter_encoder = Encoder(EncoderLayer(hid_size, c(attn), c(ff), dropout), n_layers)
        ctx_encoder = Encoder(EncoderLayer(hid_size, c(attn), c(ff), dropout), n_layers)
        decoder = Decoder(DecoderLayer(hid_size, c(attn), c(attn), c(ff), dropout), n_layers)

        # char_emb = Embeddings(char_emb_size, char_vocab)
        text_emb = nn.Sequential(ComboEmbeddings(hid_size, text_vocab, char_vocab=char_vocab),
                                 PositionalEncoding(hid_size, dropout))
        generator = Generator(hid_size, text_vocab)

        model = HieroTransformer(utter_encoder=utter_encoder,
                                 ctx_encoder=ctx_encoder,
                                 decoder=decoder,
                                 combo_embs=text_emb,
                                 generator=generator,
                                 dropout=dropout)

        # Tied embeddings
        if tied_emb:
            model.generator.proj.weight = model.combo_embs[0].text_emb.weight

        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model, args


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

        scores = x_probs.contiguous().view(-1, x_probs.size(-1))  # B x T x V --> B.T x V
        truth = y_seqs.contiguous().view(-1)  # B x T --> B.T
        assert norm != 0
        loss = self.criterion(scores, truth).sum() / norm
        if train_mode:  # dont do this for validation set
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return loss.item() * norm


class HieroTransformerTrainer(SteppedTrainer):

    def __init__(self, exp: DialogExperiment,
                 model: Optional[HieroTransformer] = None,
                 optim: str = 'ADAM',
                 **optim_args):
        super().__init__(exp, model, model_factory=HieroTransformer.make_model, optim=optim,
                         **optim_args)

        device_ids = list(range(torch.cuda.device_count()))
        log.info(f"Going to use {torch.cuda.device_count()} GPU(s) ; ids:{device_ids}")

        if len(device_ids) > 1:  # Multi GPU mode
            raise Exception("Multi GPU mode not supported yet")
        generator = self.model.generator

        criterion = LabelSmoothing(vocab_size=generator.vocab,
                                   padding_idx=PAD_IDX,
                                   smoothing=self._smoothing)

        self.loss_func = SimpleLossFunction(generator, criterion, opt=self.opt)

    def run_valid_epoch(self, data_iter: Iterator[DialogMiniBatch]):
        """
        :param data_iter: data iterator
        :return: loss value
        """
        start = time.time()
        total_tokens = 0
        total_loss = 0.0
        with tqdm(data_iter, unit='batch') as data_bar:
            for i, batch in enumerate(data_bar):
                num_toks = batch.tot_resp_toks.float().item()
                out = self.model(batch)
                # skip the BOS token in batch.y_seqs
                loss = self.loss_func(out, batch.resp_seqs, num_toks, False)
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

    def train(self, steps: int, check_point: int,
              check_pt_callback: Optional[Callable] = None,
              fine_tune=False, **args):
        log.info(f'Going to train for {steps} epochs; '
                 f'check point size:{check_point}; fine_tune={fine_tune}')
        keep_models = args.get('keep_models', 4)  # keep last _ models and delete the old

        if steps <= self.start_step:
            raise Exception(f'The model was already trained to {self.start_step} steps. '
                            f'Please increase the steps or clear the existing models')
        train_data = self.exp.get_train_data(loop_steps=steps - self.start_step,
                                             fine_tune=fine_tune)
        val_data = self.exp.get_val_data()

        train_state = TrainerState(self.model, check_point=check_point)
        train_state.train_mode(True)

        with tqdm(train_data, initial=self.start_step, total=steps, unit='batch') as data_bar:
            for batch in data_bar:
                batch: DialogMiniBatch = batch  # type annotation
                self.model.zero_grad()
                out = self.model(batch)

                num_toks = batch.tot_resp_toks.float().item()
                # skip the BOS token in  batch.y_seqs
                loss = self.loss_func(out, batch.resp_seqs, num_toks, True)
                self.tbd.add_scalars('training', {'step_loss': loss,
                                                  'learn_rate': self.opt.curr_lr},
                                     self.opt.curr_step)

                progress_msg, is_check_pt = train_state.step(num_toks, loss)
                progress_msg += f', LR={self.opt.curr_lr:g}'

                data_bar.set_postfix_str(progress_msg, refresh=False)
                del batch  # TODO: force free memory

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
    work_dir = '/Users/tg/work/phd/cs644/project/virtchar/tmp.work.transformer.simpl'
    exp = DialogExperiment(work_dir, read_only=True)
    text_vocab = len(exp.text_field)
    char_vocab = len(exp.char_field)

    steps = 2000
    check_pt = 10

    model, _ = HieroTransformer.make_model(text_vocab,
                                           char_vocab,
                                           n_layers=4,
                                           hid_size=128,
                                           ff_size=256,
                                           n_heads=4)

    trainer = HieroTransformerTrainer(exp=exp, model=model, warmup_steps=200)

    trainer.train(steps=steps, check_point=check_pt)


if __name__ == '__main__':
    __test_model__()
