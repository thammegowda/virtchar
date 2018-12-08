import abc
import time
import traceback
from collections import OrderedDict
from typing import List, Tuple, Type, Dict, Any, Optional, Iterator

import torch
from torch import nn as nn

from virtchar import DialogExperiment
from virtchar import log, device, my_tensor as tensor, debug_mode
from virtchar.model.rnn import HRED
from virtchar.tool.dataprep import PAD_TOK, BOS_TOK, EOS_TOK, \
    RawDialogReader, ChatRec, DialogMiniBatch, Dialog, Utterance
from virtchar.model.t2t import HieroTransformer

Hypothesis = Tuple[float, List[int]]
StrHypothesis = Tuple[float, str]


# TODO: simplify the generators
class GeneratorFactory(abc.ABC):

    def __init__(self, model, *args, **kwargs):
        self.model = model

    @abc.abstractmethod
    def generate_next(self, past_ys):
        pass


class Seq2SeqGenerator(GeneratorFactory):

    def __init__(self, model: HRED, batch: DialogMiniBatch):
        """
        :param model: model
        :param batch:
        """
        super().__init__(model)
        self.enc_outs, enc_hids = model.hiero_encode(batch.utters, batch.utter_lens, batch.chars,
                                                     batch.chat_ctx_idx, batch.chat_lens)
        self.dec_hids = enc_hids
        self.resp_chars = batch.resp_chars

    def generate_next(self, past_ys):
        last_ys = past_ys[:, -1]
        log_probs, self.dec_hids, _ = self.model.dec(self.enc_outs, last_ys, self.dec_hids,
                                                     self.resp_chars)
        return log_probs


class T2TGenerator(GeneratorFactory):

    def __init__(self, model: HieroTransformer, batch: DialogMiniBatch):
        super().__init__(model)
        self.memory, self.inp_mask = self.model.hiero_encode(batch.utters, batch.utter_lens,
                                                             batch.chars, batch.chat_ctx_idx)
        self.resp_chars = batch.resp_chars

    def generate_next(self, past_ys):
        # TODO: this is in efficient since the past_ys are encoded freshly for every time step
        out = self.model.decode(self.memory, self.inp_mask, past_ys, self.resp_chars)
        log_probs = self.model.generator(out[:, -1])
        return log_probs


generators = {'TRANSFORMER': T2TGenerator,
              'HRED': Seq2SeqGenerator
              }
factories = {
    'TRANSFORMER': HieroTransformer.make_model,
    'HRED': HRED.make_model,
}


class ReloadEvent(Exception):
    """An exception to reload model with new path
    -- Its a kind of hack to pass event back to caller and redo interactive shell--
    """

    def __init__(self, model_paths, state: Dict[str, Any]):
        super().__init__()
        self.model_paths = model_paths
        self.state = state


class Decoder:
    pad_val = PAD_TOK[1]
    bos_val = BOS_TOK[1]
    eos_val = EOS_TOK[1]
    default_beam_size = 5

    def __init__(self, model, gen_factory: Type[GeneratorFactory], exp: DialogExperiment,
                 gen_args=None,
                 debug=debug_mode):
        self.model = model
        self.exp: DialogExperiment = exp
        self.gen_factory = gen_factory
        self.debug = debug
        self.gen_args = gen_args if gen_args is not None else {}

    def generator(self, batch: DialogMiniBatch):
        return self.gen_factory(self.model, batch, **self.gen_args)

    @staticmethod
    def average_states(state_dict: OrderedDict, *state_dicts: OrderedDict):
        w = 1.0 / (1 + len(state_dicts))
        if state_dicts:
            key_set = set(state_dict.keys())
            assert all(key_set == set(st.keys()) for st in state_dicts)
            for key in key_set:
                state_dict[key] *= w
                for st in state_dicts:
                    state_dict[key] += w * st[key]
        return state_dict

    @staticmethod
    def _checkpt_to_model_state(checkpt_path: str):
        state = torch.load(checkpt_path, map_location=device)
        if 'model_state' in state:
            state = state['model_state']
        return state

    @staticmethod
    def maybe_ensemble_state(exp, model_paths: Optional[List[str]], ensemble: int = 1):
        if model_paths and len(model_paths) == 1:
            log.info(f" Restoring state from requested model {model_paths[0]}")
            return Decoder._checkpt_to_model_state(model_paths[0])
        elif not len(model_paths) and ensemble <= 1:
            model_path, _ = exp.get_best_known_model()
            log.info(f" Restoring state from best known model: {model_path}")
            return Decoder._checkpt_to_model_state(model_path)
        else:
            if not model_paths:
                # Average
                model_paths = exp.list_models()[:ensemble]
            log.info(f"Averaging {len(model_paths)} model states :: {model_paths}")
            states = [Decoder._checkpt_to_model_state(mp) for mp in model_paths]
            return Decoder.average_states(*states)

    @classmethod
    def new(cls, exp: DialogExperiment, model=None, gen_args=None,
            model_paths: Optional[List[str]] = None,
            ensemble: int = 1):
        """
        create a new decoder
        :param exp: experiment
        :param model: Optional pre initialized model
        :param gen_args: any optional args needed for generator
        :param model_paths: optional model paths
        :param ensemble: number of models to use for ensembling (if model is not specified)
        :return:
        """
        if model is None:
            factory = factories[exp.model_type]
            model = factory(**exp.model_args)[0]
            state = cls.maybe_ensemble_state(exp, model_paths=model_paths, ensemble=ensemble)
            model.load_state_dict(state)
            log.info("Successfully restored the model state.")
        elif isinstance(model, nn.DataParallel):
            model = model.module

        model = model.eval().to(device=device)
        generator = generators[exp.model_type]
        return cls(model, generator, exp, gen_args)

    def greedy_decode(self, mini_batch: DialogMiniBatch, max_len, **args) -> List[Hypothesis]:
        """
        Implements a simple greedy decoder
        :param mini_batch:
        :param max_len:
        :return:
        """
        gen = self.generator(mini_batch)
        batch_size = mini_batch.n_chats

        ys = torch.full(size=(batch_size, 1), fill_value=self.bos_val, dtype=torch.long,
                        device=device)
        scores = torch.zeros(batch_size, device=device)

        actives = ys[:, -1] != self.eos_val
        for i in range(1, max_len + 1):
            if actives.sum() == 0:  # all sequences Ended
                break
            log_prob = gen.generate_next(ys)
            max_prob, next_word = torch.max(log_prob, dim=1)
            scores += max_prob
            ys = torch.cat([ys, next_word.view(batch_size, 1)], dim=1)
            actives &= ys[:, -1] != self.eos_val

        result = []
        for i in range(batch_size):
            result.append((scores[i].item(), ys[i, 1:].tolist()))
        return result

    @staticmethod
    def masked_select(x, mask):
        assert x.shape[0] == mask.shape[0]
        assert mask.shape[1] == 1
        selected = x.masked_select(mask)
        return selected.view(-1, x.size(1))

    def beam_decode(self, batch: DialogMiniBatch, max_len=80, beam_size=default_beam_size,
                    num_hyp=None, skip_top=4,
                    **args) -> List[List[Hypothesis]]:
        """
        :param batch: input batch of sequences
        :param max_len:  maximum length to consider if decoder doesnt produce EOS token
        :param beam_size: beam size
        :param num_hyp: number of hypothesis in each beam to return
        :param args:
        :param skip_top: number of top beams to skip
        :return: List of num_hyp Hypothesis for each sequence in the batch.
         Each hypothesis consists of score and a list of word indexes.
        """
        # TODO: rewrite this, this function is a total mess!
        #       TG is not happy about it; it was his first attempt to implement beam decoder

        # repeat beam size
        batch_size = batch.n_chats
        assert batch_size == 1  # TODO: test large batches
        if not num_hyp:
            num_hyp = beam_size
        beam_size = max(beam_size, num_hyp)

        beamed_batch = batch.to_beamed_batch(beam_size)
        # Everything beamed_*  below is the batch repeated beam_size times
        beamed_batch_size = beamed_batch.n_chats
        assert beamed_batch_size == beam_size * batch_size

        generator = self.generator(beamed_batch)

        beamed_ys = torch.full(size=(beamed_batch_size, 1), fill_value=self.bos_val,
                               dtype=torch.long, device=device)
        beamed_scores = torch.zeros((beamed_batch_size, 1), device=device)

        beam_active = torch.ones((beamed_batch_size, 1), dtype=torch.uint8, device=device)

        # zeros means ended, one means active
        for t in range(1, max_len + 1):
            if beam_active.sum() == 0:
                break

            # ignoring top beams, to increase diversity. decrease monotonically until skip_top
            skip_t = max(skip_top - t, 0)

            # [batch*beam, Vocab]
            log_prob = generator.generate_next(beamed_ys)

            # broad cast scores along row (broadcast) and sum log probabilities
            next_scores = beamed_scores + beam_active.float() * log_prob  # Zero out inactive beams

            # [batch*beam, beam],[batch*beam, beam]
            top_scores, nxt_idx = next_scores.topk(k=beam_size + skip_t, dim=-1)

            top_scores, nxt_idx = top_scores[:, skip_t:], nxt_idx[:, skip_t:]

            # Now we got beam_size*beam_size heads, task: shrink it to beam_size
            # Since the ys will change, after re-scoring, we will make a new tensor for new ys
            new_ys = torch.full(size=(beamed_batch_size, beamed_ys.size(1) + 1),
                                fill_value=self.pad_val,
                                device=device, dtype=torch.long)

            for i in range(batch_size):
                # going to picking top k out of k*k beams for each sequence in batch
                # beams of i'th sequence in batch have this start and end
                start, end = i * beam_size, (i + 1) * beam_size
                if beam_active[start:end].sum() == 0:
                    # current sequence ended
                    new_ys[:start:end, :-1] = beamed_ys[start:end]
                    continue

                if t == 1:
                    # initialization ; since sequences are duplicated in the start, we just pick the first row
                    # it is must, otherwise, beam will select the top 1 of each beam which is same but duplicated
                    seqi_top_scores, seqi_nxt_ys = top_scores[start, :], nxt_idx[start, :]
                    # ys seen so far remains same; no reordering
                    new_ys[start:end, :-1] = beamed_ys[start:end]
                    seqi_top_scores = seqi_top_scores.view(-1, 1)
                else:
                    seqi_nxt_ys = torch.full((beam_size, 1), fill_value=self.pad_val, device=device)
                    seqi_top_scores = torch.zeros((beam_size, 1), device=device)

                    # ignore the inactive beams, don't grow them any further
                    # INACTIVE BEAMS: Preserve the inactive beams, just copy them
                    seqi_inactive_mask = (beam_active[start:end, -1] == 0).view(-1, 1)
                    seqi_inactive_count = seqi_inactive_mask.sum()
                    active_start = start + seqi_inactive_count  # [start, ... active_start-1, active_start, ... end]
                    if seqi_inactive_count > 0:  # if there are some inactive beams
                        seqi_inactive_ys = self.masked_select(beamed_ys[start:end, :],
                                                              seqi_inactive_mask)
                        new_ys[start: active_start, :-1] = seqi_inactive_ys  # Copy inactive beams
                        seqi_top_scores[start:active_start, :] = \
                            self.masked_select(beamed_scores[start:end, :], seqi_inactive_mask)

                    # ACTIVE BEAMS: the remaining beams should be let to grow
                    seqi_active_mask = (beam_active[start:end, -1] == 1).view(-1, 1)
                    seqi_active_count = seqi_active_mask.sum()  # task is to select these many top scoring beams

                    seqi_scores = self.masked_select(top_scores[start:end, :], seqi_active_mask)
                    seqi_nxt_idx = self.masked_select(nxt_idx[start:end, :], seqi_active_mask)

                    seqi_active_top_scores, seqi_nxt_idx_idx = seqi_scores.view(-1).topk(
                        k=seqi_active_count)
                    seqi_top_scores[active_start:end, 0] = seqi_active_top_scores
                    seqi_nxt_ys[active_start: end, 0] = seqi_nxt_idx.view(-1)[seqi_nxt_idx_idx]

                    # Select active ys
                    active_beam_idx = seqi_nxt_idx_idx / beam_size
                    seqi_active_ys = self.masked_select(beamed_ys[start:end, :], seqi_active_mask)
                    seqi_active_old_ys = seqi_active_ys.index_select(0, active_beam_idx)
                    new_ys[active_start:end, :-1] = seqi_active_old_ys

                    # Update status of beam_active flags
                    beam_active[active_start:end, :] = self.masked_select(beam_active[start:end, :],
                                                                          seqi_active_mask) \
                        .index_select(0, active_beam_idx)
                    if active_start > start:
                        beam_active[start:active_start, -1] = 0  # inactive beams are set to zero

                beamed_scores[start:end, :] = seqi_top_scores
                # copy the new word indices to last step
                new_ys[start:end, -1] = seqi_nxt_ys.view(-1)
            beamed_ys = new_ys
            # AND to update active flag
            beam_active = beam_active & (beamed_ys[:, -1] != self.eos_val).view(beamed_batch_size,
                                                                                1)

        result = []
        # reverse sort based on the score
        for i in range(batch_size):
            result.append([])
            start, end = i * beam_size, (i + 1) * beam_size
            scores, indices = beamed_scores[start:end, :].view(-1).sort(descending=True)
            for j in range(beam_size):
                if len(result[-1]) == num_hyp:
                    continue
                result[-1].append(
                    (scores[j].item(), beamed_ys[start + indices[j], 1:].squeeze().tolist()))
        return result

    @property
    def text_vocab(self):
        return self.exp.text_field

    @property
    def char_vocab(self):
        return self.exp.char_field

    def generate_chat(self, batch: DialogMiniBatch, beam_size, **args) -> List[StrHypothesis]:
        assert batch.n_chats == 1  # testing one at a time, for now

        if beam_size == 1:
            greedy_score, greedy_out = self.greedy_decode(batch, **args)[0]
            greedy_out = self.text_vocab.decode_ids(greedy_out, trunc_eos=True)
            log.debug(f'Greedy : score: {greedy_score:.4f} :: {greedy_out}')
            result = [(greedy_score, greedy_out)]
        else:
            assert beam_size > 1
            beams: List[List[Hypothesis]] = self.beam_decode(batch, beam_size=beam_size, **args)
            beams = beams[0]  # first sentence, the only one we passed to it as input
            result = []
            for i, (score, beam_toks) in enumerate(beams):
                out = self.text_vocab.decode_ids(beam_toks, trunc_eos=True)
                if self.debug:
                    log.debug(f"Beam {i}: score:{score:.4f} :: {out}")
                result.append((score, out))

        return result

    # noinspection PyUnresolvedReferences
    def decode_interactive(self, **args):
        import sys
        import readline
        helps = [(':quit OR :q', 'Exit'),
                 (':help OR :h', 'Print this help message'),
                 (':beam_size <n>', 'Set beam size to n'),
                 (':num_hyp <k>', 'Print top k hypotheses'),
                 (':clear OR :c', 'clear the context'),
                 (':debug', 'Flip debug mode. ON->OFF; OFF->ON'),
                 (':args', 'Print args'),
                 (':c or :clear', 'Clear context'),
                 (':models', 'show all available models of this experiment'),
                 (':model <number>', 'reload shell with the model chosen by <number>')
                 ]

        args['min_ctx'] = args.get('min_ctx', self.exp.min_ctx)
        args['max_ctx'] = args.get('max_ctx', self.exp.max_ctx)
        has_src_chars, has_tgt_chars = self.model.has_char_embs
        assert has_src_chars == has_tgt_chars, 'both on or both off' # only this is supported

        has_chars = has_src_chars
        context = []

        def print_cmds():
            for cmd, msg in helps:
                print(f"\t{cmd:15}\t-\t{msg}")

        def print_feedback(msg):
            print('\t|' + msg)

        print("Launching Interactive shell...")
        print_cmds()
        print_state = True
        tgt_char_name = None
        while True:
            if print_state:
                print_feedback('  '.join(f'{k}={v}' for k, v in args.items()))
                print_state = False


            line = input('Input: ')
            line = line.strip()
            try:
                if line in (':quit', ':q'):
                    break
                elif line in (':help', ':h'):
                    print_cmds()
                elif line in (':clear', ':c'):
                    context.clear()

                    tgt_char_name = None
                elif line.startswith(":beam_size "):
                    args['beam_size'] = int(line.replace(':beam_size', '').strip())
                    print_state = True
                elif line.startswith(":num_hyp"):
                    args['num_hyp'] = int(line.replace(':num_hyp', '').strip())
                    print_state = True
                elif line.startswith(":debug"):
                    args['debug'] = self.debug = not self.debug

                    print_state = True
                elif line.startswith(":path"):
                    self.gen_args['path'] = line.replace(':path', '').strip()
                    print_state = True
                elif line.startswith(":models"):
                    for i, mod_path in enumerate(self.exp.list_models()):
                        print(f"\t{i}\t{mod_path}")
                elif line.startswith(":model"):
                    mod_idxs = [int(x) for x in line.replace(":model", "").strip().split()]
                    models = self.exp.list_models()
                    mod_paths = []
                    for mod_idx in mod_idxs:
                        if 0 <= mod_idx < len(models):
                            mod_paths.append(str(models[mod_idx]))
                        else:
                            print(f"\tERROR: Index {mod_idx} is invalid")
                    if mod_paths:
                        print(f"\t Switching to models {mod_paths}")
                        raise ReloadEvent(mod_paths, state=args)
                else:
                    line = line.strip()
                    generate = False
                    if not line:
                        # empty line --> try to generate
                        if len(context) < args['min_ctx']:
                            print_feedback(f"At least {args['min_ctx']} should be in context."
                                           f" Currently you've {len(context)}")
                        elif has_chars and not tgt_char_name:
                            print_feedback(f"Target character name required. For example type:")
                            print_feedback(f"<char_name>:")
                        else:
                            generate = True
                    else:
                        print_update_msg = True
                        if not has_chars:
                            context.append((PAD_TOK[0], line))
                        else:
                            assert ':' in line
                            parts = line.split(':')
                            if not parts[1]: # empty text
                                tgt_char_name = parts[0]
                                print_feedback(f"Character name changed to {tgt_char_name}")
                                generate = len(context) >= args['min_ctx']
                                print_update_msg = not generate
                            else:
                                assert len(parts) > 1, 'Expected: <char_name>: <text here>'
                                char_name = parts[0]
                                utter = ':'.join(parts[1:])
                                context.append((char_name, utter))
                        if print_update_msg:
                            msg = f"Added to context, size={len(context)}."
                            if len(context) >= args['min_ctx']:
                                msg += f" Hit an empty line to generate"
                            print_feedback(msg)

                    # when its time to generate
                    if generate:
                        start = time.time()
                        context_utters = [Utterance(
                            self.char_vocab.encode_as_id(char_name),
                            self.text_vocab.encode_as_ids(text, add_eos=True),
                            raw_text=text, raw_char=char_name)
                            for char_name, text in context]
                        placeholder_response = Utterance(self.char_vocab.encode_as_id(tgt_char_name)
                                                         if tgt_char_name else PAD_TOK[1],
                                                         text=[EOS_TOK[1]], raw_text='',
                                                         raw_char=tgt_char_name)
                        rec = ChatRec(context_utters, response=placeholder_response)
                        batch = rec.as_dialog_mini_batch(sort_dec=False)

                        print("Context:")
                        for cn, txt in context:
                            print(f'<< {cn if cn != PAD_TOK[0] else ""} : {txt}')
                        print("Response:")
                        result = self.generate_chat(batch, **args)
                        pref = f"{tgt_char_name}:" if has_chars and tgt_char_name else ''
                        for score, hyp in result:
                            print(f' >> {pref} {hyp}\t{score:.4f}')
                        print_feedback(f'took={1000 * (time.time()-start):.3f}ms')
                        context.append((tgt_char_name if tgt_char_name else PAD_TOK[0], result[0][1]))
            except ReloadEvent as re:
                raise re  # send it to caller
            except EOFError as e1:
                break
            except Exception:
                traceback.print_exc()
                print_state = True

    def decode_dialogs(self, dialogs: Iterator[Dialog], out, **args):
        min_ctx, max_ctx = self.exp.min_ctx, self.exp.max_ctx
        test_chars = None
        for i, dialog in enumerate(dialogs):
            chats: Iterator[ChatRec] = dialog.as_test_chats(min_ctx=min_ctx, max_ctx=max_ctx,
                                                            test_chars=test_chars)
            for j, chat in enumerate(chats):
                # One chat in batch. Should/can be improved later
                batch = chat.as_dialog_mini_batch()
                log.info(f"dialog: {i}: chat: {j} :: \n"
                         f"MSG: {chat.context[-1].raw_char}: {chat.context[-1].raw_text}\n"
                         f"RSP: {chat.response.raw_char}: {chat.response.raw_text}")

                result = self.generate_chat(batch, **args)
                num_hyp = args['num_hyp']
                out_line = '\n'.join(f'{hyp}\t{score:.4f}' for score, hyp in result) + '\n'
                log.info(f"OUT:\n{out_line}")
                if out:
                    out.write(out_line)
                    if num_hyp > 1:
                        out.write('\n')

    def decode_file(self, inp, out, **args):
        reader = RawDialogReader(inp, text_field=self.text_vocab, char_field=self.char_vocab)
        self.decode_dialogs(reader, out, **args)
