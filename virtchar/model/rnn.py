import random
from typing import Optional, Callable, Iterator, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from tqdm import tqdm

from virtchar import log, DialogExperiment as Experiment
from virtchar import my_tensor as tensor, device
from virtchar.tool.dataprep import PAD_TOK_IDX, BOS_TOK_IDX, DialogMiniBatch
from virtchar.model import DialogModel
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


class BaseEncoder(nn.Module):

    def __init__(self, inp_size, out_size, n_layers, bidirectional, dropout=0.5, rnn_type=nn.GRU):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.inp_size = inp_size
        self.out_size = out_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        out_size = self.out_size
        if self.bidirectional:
            assert self.out_size % 2 == 0
            out_size = out_size // 2
        self.rnn_type = rnn_type
        self.rnn_node = rnn_type(self.inp_size, out_size, num_layers=self.n_layers,
                                 bidirectional=self.bidirectional, batch_first=True,
                                 dropout=dropout if n_layers > 1 else 0)

    def forward(self, embedded, seq_lens, hidden=None):
        # assert not args     # must be empty
        # assert not kwargs  # it must be empty
        embedded = self.dropout(embedded)
        # sorting
        try:
            packed = nn.utils.rnn.pack_padded_sequence(embedded, seq_lens, batch_first=True)
        except:
            print(seq_lens)
            raise
        outputs, hidden = self.rnn_node(packed, hidden)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True,
                                                                   padding_value=PAD_TOK_IDX)
        return outputs, hidden

    def get_last_layer_last_step(self, enc_state) -> Tuple[Tensor, Optional[Tensor]]:
        # get the last layer's last time step output
        # lnhn is layer n hidden n which is last layer last hidden. similarly lncn
        hns, cns = enc_state if self.has_context_gate else (enc_state, None)

        # cns (i.e. context gate can be missing, for example GRU doesnt have context out)
        lncn = None

        if self.bidirectional:
            # cat bidirectional
            lnhn = hns.view(self.n_layers, 2, hns.shape[1], hns.shape[-1])[-1]
            lnhn = torch.cat([lnhn[0], lnhn[1]], dim=1)

            if self.has_context_gate:
                lncn = cns.view(self.n_layers, 2, cns.shape[1], cns.shape[-1])[-1]
                lncn = torch.cat([lncn[0], lncn[1]], dim=1)
        else:
            lnhn = hns.view(self.n_layers, hns.shape[1], hns.shape[-1])[-1]
            if self.has_context_gate:
                lncn = cns.view(self.n_layers, cns.shape[1], cns.shape[-1])[-1]
        return lnhn, lncn

    @property
    def has_context_gate(self):
        return self.rnn_type == nn.LSTM


class UtteranceEncoder(BaseEncoder):
    # also read it as SentenceEncoder

    def __init__(self, text_emb: Embedder,
                 char_emb: Embedder,
                 out_size: int,
                 n_layers: int,
                 bidirectional: bool = True,
                 dropout=0.2,
                 rnn_type=nn.GRU):
        self.emb_size = text_emb.emb_size + char_emb.emb_size
        super().__init__(inp_size=self.emb_size, out_size=out_size, n_layers=n_layers,
                         bidirectional=bidirectional, dropout=dropout, rnn_type=rnn_type)

        # First level, deals with text; so embeddings are needed
        self.text_emb = text_emb
        self.char_emb = char_emb

        if self.has_context_gate:
            # LSTMs produce h_t and c_t, we are going to compress them
            self.compressor = nn.Linear(self.out_size * 2, self.out_size)

    def forward(self, utters: Tensor, utter_lens: Tensor, chars: Tensor = None, hidden=None):
        batch_size = utters.shape[0]
        if chars is not None:
            assert batch_size == chars.shape[0]
        seq_len = utters.shape[1]  # padded

        text_embedded = self.text_emb(utters).view(batch_size, seq_len, self.text_emb.emb_size)
        char_embedded = self.char_emb(chars).view(batch_size, 1, self.char_emb.emb_size)
        # repeat the character embedding for each time step in sequence
        char_embedded = char_embedded.expand(batch_size, seq_len, self.char_emb.emb_size)

        # TODO: character embedding masking by length, when unequal length batch
        # concat word embedding and character embedding along the last dimension
        embedded = torch.cat([text_embedded, char_embedded], dim=-1)

        outputs, hidden_state = super().forward(embedded, utter_lens, hidden=hidden)

        # Sum bidirectional outputs
        # outputs = outputs[:, :, :self.hid_size] + outputs[:, :, self.hid_size:]
        lnhn, lncn = self.get_last_layer_last_step(hidden_state)
        if self.has_context_gate:
            sent_repr = self.compressor(torch.cat([lnhn, lncn], dim=-1))
        else:
            sent_repr = lnhn
        return outputs, sent_repr


class ContextEncoder(BaseEncoder):
    # also read it as ParagraphEncoder

    def __init__(self,
                 inp_size: int,
                 out_size: int,
                 n_layers: int,
                 bidirectional: bool = True,
                 dec_layers: Optional[int] = None,
                 dropout=0.2,
                 rnn_type=nn.GRU):
        super().__init__(inp_size=inp_size, out_size=out_size, n_layers=n_layers,
                         bidirectional=bidirectional, dropout=dropout,
                         rnn_type=rnn_type)
        self.dec_layers = dec_layers if dec_layers else n_layers

    def forward(self, embedded: Tensor, seq_lens: Tensor, hidden=None):
        outputs, hidden_state = super().forward(embedded, seq_lens, hidden=hidden)
        lnhn, lncn = self.get_last_layer_last_step(hidden_state)

        # duplicate it decoder layers
        lnhn = lnhn.expand(self.dec_layers, *lnhn.shape).contiguous()
        if self.has_context_gate:
            lncn = lncn.expand(self.dec_layers, *lncn.shape).contiguous()
            return outputs, (lnhn, lncn)
        else:
            assert lncn is None
            return outputs, lnhn


class SeqDecoder(nn.Module):

    def __init__(self, text_emb: Embedder, char_emb: Embedder, generator: Generator,
                 n_layers: int, dropout=0.5, rnn_type=nn.GRU):
        super(SeqDecoder, self).__init__()
        self.text_emb = text_emb
        self.char_emb = char_emb
        self.dropout = nn.Dropout(dropout)
        self.generator = generator
        self.n_layers = n_layers
        self.emb_size = self.text_emb.emb_size + self.char_emb.emb_size
        self.hid_size = self.generator.vec_size
        self.rnn_type = rnn_type
        self.rnn_node = rnn_type(self.emb_size, self.hid_size,
                                 num_layers=self.n_layers,
                                 bidirectional=False, batch_first=True,
                                 dropout=dropout if n_layers > 1 else 0)

    def embed_prev(self, prev_out, chars):
        # Get the embedding of the current input word (last output word)
        batch_size = prev_out.size(0)

        # S=B x 1 x N
        text_embedded = self.text_emb(prev_out).view(batch_size, 1, self.text_emb.emb_size)
        char_embedded = self.char_emb(chars).view(batch_size, 1, self.char_emb.emb_size)
        embedded = torch.cat([text_embedded, char_embedded], dim=-1)
        embedded = self.dropout(embedded)
        return embedded

    def forward(self, enc_outs, prev_out, last_hidden, chars: Tensor = None, gen_probs=True):
        # Note: we run this one step at a time
        assert len(enc_outs) == len(prev_out)
        # Get the embedding of the current input word (last output word)
        embedded = self.embed_prev(prev_out=prev_out, chars=chars)

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


class DotAttn(nn.Module):
    """
    Attention model
    """

    def __init__(self, hid_size):
        super(DotAttn, self).__init__()
        self.inp_size = hid_size
        self.out_size = hid_size

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

    def __init__(self, text_emb: Embedder, char_emb: Embedder, generator: Generator, n_layers: int,
                 dropout: float = 0.5, attention=None, rnn_type=nn.GRU):
        super(AttnSeqDecoder, self).__init__(text_emb=text_emb, char_emb=char_emb,
                                             generator=generator, n_layers=n_layers,
                                             dropout=dropout, rnn_type=rnn_type)
        self.attn_type = attention

        assert attention == 'dot'  # only dot is supported at the moment
        self.attn = DotAttn(self.hid_size)
        self.merge = nn.Linear(self.hid_size + self.attn.out_size, self.hid_size)

    def forward(self, enc_outs, prev_out, last_hidden, chars: Tensor = None, gen_probs=True):
        # Note: we run this one step at a time
        assert len(enc_outs) == len(prev_out)
        # Get the embedding of the current input word (last output word)
        embedded = self.embed_prev(prev_out=prev_out, chars=chars)

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


class HRED(DialogModel):

    def __init__(self, utter_enc: UtteranceEncoder, ctx_enc: ContextEncoder, dec: SeqDecoder):
        # Slight change in terminology: sentence encoder is utterance encoder, and para encoder
        """
        Hierarchical encoder decoder model
        :param utter_enc: sentence encoder
        :param ctx_enc: paragraph encoder
        :param dec:
        """
        super(HRED, self).__init__()
        self.utter_enc = utter_enc
        self.ctx_enc = ctx_enc
        self.dec = dec
        assert utter_enc.out_size == ctx_enc.inp_size
        assert ctx_enc.out_size == dec.hid_size
        self._model_dim = dec.hid_size

    @property
    def model_dim(self):
        return self._model_dim

    @property
    def has_char_embs(self) -> Tuple[bool, bool]:
        raise NotImplementedError()

    def hiero_encode(self, utters, utter_lens, chars, chat_ctx_idx, chat_lens):
        # :: level 1
        sent_outs, sent_reprs = self.utter_enc(utters, utter_lens, chars=chars, hidden=None)
        assert 2 == len(sent_reprs.shape)  # its a 2d tensor

        # Inserting 0s at the 0th row, so we can pick rows based on indices, where index=0 is pad
        padded_sent_repr = torch.cat(
            [torch.zeros(1, sent_reprs.shape[1], device=device), sent_reprs], dim=0)

        # index_select works with vector, so we flatten and then restore
        chat_input = torch.index_select(padded_sent_repr, 0, chat_ctx_idx.view(-1))
        chat_input = chat_input.view(*chat_ctx_idx.shape, -1)

        # :: level 2
        ctx_outs, dec_hids = self.ctx_enc(chat_input, chat_lens, hidden=None)
        return ctx_outs, dec_hids

    def forward(self, batch: DialogMiniBatch):
        ctx_outs, dec_hids = self.hiero_encode(batch.utters, batch.utter_lens, batch.chars,
                                               batch.chat_ctx_idx, batch.chat_lens)

        batch_size = batch.n_chats
        dec_inps = tensor([[BOS_TOK_IDX]] * batch_size, dtype=torch.long)

        # assumptions:
        #   1) the batch.resp_seq[0, :] does not start with <bos>
        #   2) the batch.resp_seq ends with <eos>
        outp_probs = torch.zeros((batch_size, batch.max_resp_len), device=device)

        for t in range(0, batch.max_resp_len):

            word_probs, dec_hids, _ = self.dec(ctx_outs, dec_inps, dec_hids, chars=batch.resp_chars)

            # expected output;; log probability for these indices should be high
            expct_word_idx = batch.resp_seqs[:, t].view(batch_size, 1)
            expct_word_log_probs = word_probs.gather(dim=1, index=expct_word_idx)
            outp_probs[:, t] = expct_word_log_probs.squeeze()

            # Randomly switch between gold and the prediction next word
            if random.choice((False, True)):
                dec_inps = expct_word_idx  # Next input is current target
            else:
                pred_word_idx = word_probs.argmax(dim=1)
                dec_inps = pred_word_idx.view(batch_size, 1)
        return outp_probs

    @staticmethod
    def make_model(text_vocab: int, char_vocab: int, text_emb_size: int = 300, char_emb_size=50,
                   hid_size: int = 300, n_layers: int = 2, dropout=0.2, attention='dot',
                   rnn_type='GRU'):
        args = {
            'text_vocab': text_vocab,
            'char_vocab': char_vocab,
            'text_emb_size': text_emb_size,
            'char_emb_size': char_emb_size,
            'hid_size': hid_size,
            'n_layers': n_layers,
            'dropout': dropout,
            'attention': attention,
            'rnn_type': rnn_type
        }
        log.info(f"RNN type={rnn_type}")
        rnn_type = {
            'GRU': nn.GRU,
            'LSTM': nn.LSTM
        }[rnn_type]
        text_embedder = Embedder('text', text_vocab, text_emb_size)
        char_embedder = Embedder('char', char_vocab, char_emb_size)
        text_generator = Generator('text', vec_size=hid_size, vocab_size=text_vocab)

        utter_enc = UtteranceEncoder(text_embedder, char_embedder, hid_size, n_layers=n_layers,
                                     bidirectional=True, dropout=dropout, rnn_type=rnn_type)
        chat_enc = ContextEncoder(utter_enc.out_size, hid_size, n_layers=n_layers,
                                  bidirectional=True, dropout=dropout, rnn_type=rnn_type)

        if attention:
            log.info(f"Using attention={attention} decoder")
            dec = AttnSeqDecoder(text_embedder, char_embedder, text_generator,
                                 n_layers=n_layers, dropout=dropout, attention=attention,
                                 rnn_type=rnn_type)
        else:
            log.info("NOT Using attention model in decoder")
            dec = SeqDecoder(text_embedder, char_embedder, text_generator, n_layers=n_layers,
                             dropout=dropout, rnn_type=rnn_type)

        model = HRED(utter_enc, chat_enc, dec)
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

    def __call__(self, log_probs, batch: DialogMiniBatch, train_mode: bool) -> float:
        per_tok_loss = -log_probs

        tok_mask = self.sequence_mask(batch.resp_lens, batch.max_resp_len)
        norm = batch.tot_resp_toks.float()
        loss = (per_tok_loss * tok_mask.float()).sum().float() / norm
        if train_mode:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        return loss.item() * norm


class SteppedHREDTrainer(SteppedTrainer):
    """
    A trainer for HRED model
    based on Optimizer steps (not the epochs)
    """

    def __init__(self, exp: Experiment,
                 model: Optional[HRED] = None,
                 optim: str = 'ADAM',
                 **optim_args):
        super().__init__(exp, model, model_factory=HRED.make_model, optim=optim, **optim_args)
        self.loss_func = SimpleLossFunction(optim=self.opt)

    def run_valid_epoch(self, data_iter: Iterator[DialogMiniBatch]) -> float:
        state = TrainerState(self.model, -1)
        with tqdm(enumerate(data_iter), unit='batch', leave=True, dynamic_ncols=True) as data_bar:
            for i, batch in data_bar:
                # Step clear gradients
                self.model.zero_grad()
                # Step Run forward pass.
                outp_log_probs = self.model(batch)
                loss = self.loss_func(outp_log_probs, batch, train_mode=False)
                bar_msg, _ = state.step(batch.tot_resp_toks.item(), loss)
                data_bar.set_postfix_str(bar_msg, refresh=False)
        return state.running_loss()

    def train(self, steps: int, check_point: int, fine_tune=False,
              check_pt_callback: Optional[Callable] = None, **args):
        log.info(f'Going to train for {steps} steps; '
                 f'check point size:{check_point}; fine tune={fine_tune}')
        keep_models = args.get('keep_models', 4)  # keep last _ models and delete the old

        if steps <= self.start_step:
            raise Exception(f'The model was already trained to {self.start_step} steps. '
                            f'Please increase the steps or clear the existing models')
        train_data = self.exp.get_train_data(loop_steps=steps - self.start_step)
        val_data = self.exp.get_val_data()

        train_state = TrainerState(self.model, check_point=check_point)
        train_state.train_mode(True)
        with tqdm(train_data, initial=self.start_step, total=steps, unit='batch') as data_bar:
            for batch in data_bar:
                # Step clear gradients
                self.model.zero_grad()

                # Step Run forward pass.
                outp_log_probs = self.model(batch)
                loss = self.loss_func(outp_log_probs, batch, True)
                self.tbd.add_scalars('training', {'step_loss': loss,
                                                  'learn_rate': self.opt.curr_lr},
                                     self.opt.curr_step)
                bar_msg, is_check_pt = train_state.step(batch.tot_resp_toks.item(), loss)
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
    work_dir = '/Users/tg/work/phd/cs644/project/virtchar/tmp.work'
    exp = Experiment(work_dir, read_only=True)
    text_vocab = len(exp.text_field)
    char_vocab = len(exp.char_field)

    emb_size = 100
    char_emb_size = 50

    step_size = 50
    model_dim = 100

    steps = 2000
    check_pt = 10

    log.info(f"====== VOCAB={text_vocab}, Characters:{char_vocab}======")
    model, args = HRED.make_model(text_vocab=text_vocab, char_vocab=char_vocab,
                                  text_emb_size=emb_size, char_emb_size=char_emb_size,
                                  hid_size=model_dim, n_layers=1)
    trainer = SteppedHREDTrainer(exp=exp, model=model, lr=0.01, warmup_steps=500)
    trainer.train(steps=steps, step_size=step_size, check_point=check_pt)


if __name__ == '__main__':
    __test_seq2seq_model__()
