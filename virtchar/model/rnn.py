import random
from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from virtchar import log, TranslationExperiment as Experiment
from virtchar import my_tensor as tensor, device
from virtchar.tool.dataprep import PAD_TOK_IDX, BOS_TOK_IDX, Batch, BatchIterable
from virtchar.model import NMTModel
from virtchar.model.trainer import TrainerState, SteppedTrainer


class Embedder(nn.Embedding):
    """
    This module takes words (word IDs, not the text ) and creates vectors.
    For the inverse operation see  `Generator` module
    """

    def __init__(self, name: str, vocab_size: int, emb_size: int,
                 weights: Optional[torch.Tensor] = None):
        self.name = name
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        super(Embedder, self).__init__(self.vocab_size, self.emb_size, padding_idx=PAD_TOK_IDX,
                                       _weight=weights)


class Generator(nn.Module):
    """
    This module takes vectors and produces word ids.
    Note: In theory, this is not inverse of `Embedder`, however in practice it is an approximate
    inverse operation of `Embedder`
    """

    def __init__(self, name: str, vec_size: int, vocab_size: int):
        super(Generator, self).__init__()
        self.name = name
        self.vec_size = vec_size
        self.vocab_size = vocab_size
        self.proj = nn.Linear(self.vec_size, self.vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class SeqEncoder(nn.Module):

    def __init__(self, embedder: Embedder, out_size: int, n_layers: int,
                 bidirectional: bool = True, dropout=0.5):
        super().__init__()
        self.emb = embedder
        self.dropout = nn.Dropout(dropout)
        self.emb_size = self.emb.emb_size
        self.out_size = out_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        out_size = self.out_size
        if self.bidirectional:
            assert self.out_size % 2 == 0
            out_size = out_size // 2
        self.rnn_node = nn.LSTM(self.emb_size, out_size, num_layers=self.n_layers,
                                bidirectional=self.bidirectional, batch_first=True,
                                dropout=dropout if n_layers > 1 else 0)

    def forward(self, input_seqs: torch.Tensor, input_lengths, hidden=None, pre_embedded=False):
        assert len(input_seqs) == len(input_lengths)
        if pre_embedded:
            embedded = input_seqs
            batch_size, seq_len, emb_size = input_seqs.shape
            assert emb_size == self.emb_size
        else:
            batch_size, seq_len = input_seqs.shape
            embedded = self.emb(input_seqs).view(batch_size, seq_len, self.emb_size)
        embedded = self.dropout(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.rnn_node(packed, hidden)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True,
                                                                   padding_value=PAD_TOK_IDX)
        # Sum bidirectional outputs
        # outputs = outputs[:, :, :self.hid_size] + outputs[:, :, self.hid_size:]
        return outputs, self.to_dec_state(hidden)

    def to_dec_state(self, enc_state):
        # get the last layer's last time step output
        # lnhn is layer n hidden n which is last layer last hidden. similarly lncn
        hns, cns = enc_state
        if self.bidirectional:
            # cat bidirectional
            lnhn = hns.view(self.n_layers, 2, hns.shape[1], hns.shape[-1])[-1]
            lnhn = torch.cat([lnhn[0], lnhn[1]], dim=1)
            lncn = cns.view(self.n_layers, 2, cns.shape[1], cns.shape[-1])[-1]
            lncn = torch.cat([lncn[0], lncn[1]], dim=1)
        else:
            lnhn = hns.view(self.n_layers, hns.shape[1], hns.shape[-1])[-1]
            lncn = cns.view(self.n_layers, cns.shape[1], cns.shape[-1])[-1]

        # lnhn and lncn hold compact representation
        # duplicate for decoder layers
        return (lnhn.expand(self.n_layers, *lnhn.shape).contiguous(),
                lncn.expand(self.n_layers, *lncn.shape).contiguous())


class SeqDecoder(nn.Module):

    def __init__(self, prev_emb_node: Embedder, generator: Generator, n_layers: int, dropout=0.5):
        super(SeqDecoder, self).__init__()
        self.prev_emb = prev_emb_node
        self.dropout = nn.Dropout(dropout)
        self.generator = generator
        self.n_layers = n_layers
        self.emb_size = self.prev_emb.emb_size
        self.hid_size = self.generator.vec_size
        self.rnn_node = nn.LSTM(self.emb_size, self.hid_size, num_layers=self.n_layers,
                                bidirectional=False, batch_first=True,
                                dropout=dropout if n_layers > 1 else 0)

    def forward(self, enc_outs, prev_out, last_hidden, gen_probs=True):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = prev_out.size(0)
        assert len(enc_outs) == batch_size
        # S=B x 1 x N
        embedded = self.prev_emb(prev_out).view(batch_size, 1, self.prev_emb.emb_size)
        embedded = self.dropout(embedded)
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.rnn_node(embedded, last_hidden)

        # [B x N ] <- [B x S=1 x N]
        rnn_output = rnn_output.squeeze(1)

        if gen_probs:
            # Finally predict next token
            next_word_distr = self.generator(rnn_output)
            # Return final output, hidden state, and attention weights (for visualization)
            return next_word_distr, hidden, None
        else:
            return rnn_output, hidden, None


class GeneralAttn(nn.Module):
    """
    Attention model
    """

    def __init__(self, hid_size):
        super(GeneralAttn, self).__init__()
        self.inp_size = hid_size
        self.out_size = hid_size
        self.attn = nn.Linear(self.inp_size, self.out_size)

    def forward(self, this_rnn_out, encoder_outs):
        # hidden      : [B, D]
        # encoder_out : [B, S, D]
        #    [B, D] --> [B, S, D] ;; repeat hidden sequence_len times
        this_run_out = this_rnn_out.unsqueeze(1).expand_as(encoder_outs)
        #  A batched dot product implementation using element wise product followed by sum
        #    [B, S]  <-- [B, S, D]
        # element wise multiply, then sum along the last dim (i.e. model_dim)
        weights = (encoder_outs * this_run_out).sum(dim=-1)

        # Normalize energies to weights in range 0 to 1
        return F.softmax(weights, dim=1)


class AttnSeqDecoder(SeqDecoder):
    def __init__(self, prev_emb_node: Embedder, generator: Generator, n_layers: int,
                 dropout: float=0.5):
        super(AttnSeqDecoder, self).__init__(prev_emb_node, generator, n_layers, dropout=dropout)
        self.attn = GeneralAttn(self.hid_size)
        self.merge = nn.Linear(self.hid_size + self.attn.out_size, self.hid_size)

    def forward(self, enc_outs, prev_out, last_hidden, gen_probs=True):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = prev_out.size(0)
        embedded = self.prev_emb(prev_out)
        embedded = embedded.view(batch_size, 1, self.prev_emb.emb_size)
        embedded = self.dropout(embedded)
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.rnn_node(embedded, last_hidden)
        # [B x N ] <- [B x S=1 x  N]
        rnn_output = rnn_output.squeeze(1)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, enc_outs)  # B x S
        #   attn_weights : B x S     --> B x 1 x S
        #   enc_outs     : B x S x N
        # Batch multiply : [B x 1 x S] [B x S x N] --> [B x 1 x N]
        context = attn_weights.unsqueeze(1).bmm(enc_outs)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        # rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.merge(concat_input))

        if gen_probs:
            # predict next token
            output_probs = self.generator(concat_output)
            # Return final output, hidden state, and attention weights (for visualization)
            return output_probs, hidden, attn_weights
        else:
            return concat_output, hidden, attn_weights


class Seq2Seq(NMTModel):

    def __init__(self, enc: SeqEncoder, dec: SeqDecoder):
        super(Seq2Seq, self).__init__()
        self.enc = enc
        self.dec = dec
        assert enc.out_size == dec.hid_size

    @property
    def model_dim(self):
        return self.enc.out_size

    def encode(self, x_seqs, x_lens, hids=None):
        enc_outs, enc_hids = self.enc(x_seqs, x_lens, hids)
        return enc_outs, enc_hids

    def forward(self, batch: Batch):
        assert batch.batch_first
        batch_size = len(batch)
        enc_outs, enc_hids = self.encode(batch.x_seqs, batch.x_len, hids=None)

        dec_inps = tensor([[BOS_TOK_IDX]] * batch_size, dtype=torch.long)
        dec_hids = enc_hids
        """
        # extract vector at given last stamp (as per the seq length)
        t_dim = 1
        lastt_idx = (batch.x_len - 1).view(-1, 1).expand(-1, self.enc.out_size).unsqueeze(t_dim)
        lastt_out = enc_outs.gather(dim=t_dim, index=lastt_idx).squeeze(t_dim)
        lastt_out = lastt_out.expand(self.dec.n_layers, batch_size, self.dec.generator.vec_size)
        dec_hids = (lastt_out, lastt_out)   # copy enc output to h and c of LSTM
        """
        outp_probs = torch.zeros((batch.max_y_len - 1, batch_size), device=device)

        for t in range(1, batch.max_y_len):
            word_probs, dec_hids, _ = self.dec(enc_outs, dec_inps, dec_hids)

            # expected output;; log probability for these indices should be high
            expct_word_idx = batch.y_seqs[:, t].view(batch_size, 1)
            expct_word_log_probs = word_probs.gather(dim=1, index=expct_word_idx)
            outp_probs[t - 1] = expct_word_log_probs.squeeze()

            # Randomly switch between gold and the prediction next word
            if random.choice((False, True)):
                dec_inps = expct_word_idx  # Next input is current target
            else:
                pred_word_idx = word_probs.argmax(dim=1)
                dec_inps = pred_word_idx.view(batch_size, 1)
        return outp_probs.t()

    @staticmethod
    def make_model(src_lang, tgt_lang, src_vocab: int, tgt_vocab: int, emb_size: int = 300,
                   hid_size: int = 300, n_layers: int = 2, attention=False, dropout=0.33,
                   tied_emb: Optional[str] = None):
        args = {
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
            'emb_size': emb_size,
            'hid_size': hid_size,
            'n_layers': n_layers,
            'attention': attention,
            'dropout': dropout,
            'tied_emb': tied_emb
        }
        src_embedder = Embedder(src_lang, src_vocab, emb_size)
        tgt_embedder = Embedder(tgt_lang, tgt_vocab, emb_size)
        tgt_generator = Generator(tgt_lang, vec_size=hid_size, vocab_size=tgt_vocab)
        if tied_emb:
            assert src_vocab == tgt_vocab
            if tied_emb == 'three-way':
                log.info('Tying embedding three way : SrcIn == TgtIn == TgtOut')
                src_embedder.weight = tgt_embedder.weight
                tgt_generator.proj.weight = tgt_embedder.weight
            elif tied_emb == 'two-way':
                log.info('Tying embedding two way : SrcIn == TgtIn')
                src_embedder.weight = tgt_embedder.weight
            else:
                raise Exception('Invalid argument to tied_emb; Known: {three-way, two-way}')

        enc = SeqEncoder(src_embedder, hid_size, n_layers=n_layers, bidirectional=True,
                         dropout=dropout)
        if attention:
            log.info("Using attention models for decoding")
            dec = AttnSeqDecoder(tgt_embedder, tgt_generator, n_layers=n_layers, dropout=dropout)
        else:
            log.info("NOT Using attention models for decoding")
            dec = SeqDecoder(tgt_embedder, tgt_generator, n_layers=n_layers, dropout=dropout)

        model = Seq2Seq(enc, dec)
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model, args


def aeq(*items):
    for i in items[1:]:
        if items[0] != i:
            return False
    return True


class SimpleLossFunction:

    def __init__(self, optim):
        self.optim = optim

    @staticmethod
    def sequence_mask(lengths, max_len):
        batch_size = lengths.size(0)
        # create a row [0, 1, ... s] and duplicate this row batch_size times --> [B, S]
        seq_range_expand = torch.arange(0, max_len, dtype=torch.long,
                                        device=device).expand(batch_size, max_len)
        # make lengths vectors to [B x 1] and duplicate columns to [B, S]
        seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand < seq_length_expand  # 0 if padding, 1 otherwise

    def __call__(self, log_probs, batch: Batch, train_mode: bool) -> float:
        per_tok_loss = -log_probs

        tok_mask = self.sequence_mask(batch.y_len, batch.max_y_len - 1)
        norm = batch.y_toks
        loss = (per_tok_loss * tok_mask.float()).sum().float() / norm
        if train_mode:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        return loss.item() * norm


class SteppedSeq2SeqTrainer(SteppedTrainer):

    def __init__(self, exp: Experiment,
                 model: Optional[Seq2Seq] = None,
                 optim: str = 'ADAM',
                 **optim_args):
        super().__init__(exp, model, model_factory=Seq2Seq.make_model, optim=optim, **optim_args)
        self.loss_func = SimpleLossFunction(optim=self.opt)

    def run_valid_epoch(self, data_iter: BatchIterable) -> float:
        state = TrainerState(self.model, -1)
        with tqdm(data_iter, total=data_iter.num_batches, unit='batch') as data_bar:
            for i, batch in enumerate(data_bar):
                batch = batch.to(device)
                # Step clear gradients
                self.model.zero_grad()
                # Step Run forward pass.
                outp_log_probs = self.model(batch)
                loss = self.loss_func(outp_log_probs, batch, train_mode=False)
                bar_msg, _ = state.step(batch.y_toks, loss)
                data_bar.set_postfix_str(bar_msg, refresh=False)
                del batch
        return state.running_loss()

    def train(self, steps: int, check_point: int, batch_size: int, fine_tune=False,
              check_pt_callback: Optional[Callable] = None, **args):
        log.info(f'Going to train for {steps} steps; batch_size={batch_size}; '
                 f'check point size:{check_point}; fine tune={fine_tune}')
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
                # Step clear gradients
                self.model.zero_grad()
                # Step Run forward pass.
                outp_log_probs = self.model(batch)

                loss = self.loss_func(outp_log_probs, batch, True)
                self.tbd.add_scalars('training', {'step_loss': loss,
                                                  'learn_rate': self.opt.curr_lr},
                                     self.opt.curr_step)
                bar_msg, is_check_pt = train_state.step(batch.y_toks, loss)
                bar_msg += f', LR={self.opt.curr_lr:g}'
                data_bar.set_postfix_str(bar_msg, refresh=False)

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


def __test_seq2seq_model__():
    from virtchar.tool.dummy import DummyExperiment
    from virtchar.use.decoder import Decoder

    vocab_size = 20
    batch_size = 30
    exp = DummyExperiment("tmp.work", config={'model_type': 'seq'
                                                            '2seq'},
                          read_only=True, vocab_size=vocab_size)
    emb_size = 100
    model_dim = 100
    steps = 2000
    check_pt = 100

    assert 2 == Batch.bos_val
    src = tensor([[2,  4,  5,  6,  7, 8, 9, 10, 11, 12, 13],
                  [2, 13, 12, 11, 10, 9, 8,  7,  6,  5,  4]])
    src_lens = tensor([src.size(1)] * src.size(0))

    for reverse in (False,):
        # train two models;
        #  first, just copy the numbers, i.e. y = x
        #  second, reverse the numbers y=(V + reserved - x)
        log.info(f"====== REVERSE={reverse}; VOCAB={vocab_size}======")
        model, args = Seq2Seq.make_model('DummyA', 'DummyB', vocab_size, vocab_size, attention=True,
                                         emb_size=emb_size, hid_size=model_dim, n_layers=1)
        trainer = SteppedSeq2SeqTrainer(exp=exp, model=model, lr=0.01, warmup_steps=500)
        decr = Decoder.new(exp, model)

        def check_pt_callback(**args):
            res = decr.greedy_decode(src, src_lens, max_len=17)
            for score, seq in res:
                log.info(f'{score:.4f} :: {seq}')

        trainer.train(steps=steps, check_point=check_pt, batch_size=batch_size,
                      check_pt_callback=check_pt_callback)


if __name__ == '__main__':
    __test_seq2seq_model__()


