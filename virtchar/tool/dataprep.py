import os
from typing import List, Iterator, Tuple, Union, Optional
import torch
from virtchar import log
from virtchar import my_tensor as tensor, device
from virtchar.utils import IO
from typing import Mapping, List, Union, Set, TextIO
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from pathlib import Path
import sys
import copy
from dataclasses import dataclass, field
import random

PAD_TOK = '<pad>', 0
UNK_TOK = '<unk>', 1
BOS_TOK = '<s>', 2
EOS_TOK = '</s>', 3
CLS_TOK = '<cls>', 4

BOS_TOK_IDX = BOS_TOK[1]
EOS_TOK_IDX = EOS_TOK[1]
PAD_TOK_IDX = PAD_TOK[1]
UNK_TOK_IDX = UNK_TOK[1]
CLS_TOK_IDX = CLS_TOK[1]

RESERVED_TOKS = [PAD_TOK, UNK_TOK, BOS_TOK, EOS_TOK]

RawRecord = Tuple[str, str]
TokRawRecord = Tuple[List[str], List[str]]
MonoSeqRecord = List[Union[int, str]]
ParallelSeqRecord = Tuple[MonoSeqRecord, MonoSeqRecord]
TokStream = Union[Iterator[Iterator[str]], Iterator[str]]

CharacterField = Union[int, str]
TextField = List[Union[int, str]]
DialogRecord = Tuple[CharacterField, TextField]


class Field(SentencePieceProcessor):
    """A wrapper class for sentence piece trainer and processor"""

    def __init__(self, path: Union[str, Path]):
        super().__init__()
        if type(path) is not str:
            path = str(path)
        assert self.load(path)

    def encode_as_ids(self, text: str, add_bos=False, add_eos=False, add_cls=False) -> List[int]:
        assert not (add_bos and add_cls), 'Either BOS or CLS can be added, not both'
        ids = super(Field, self).encode_as_ids(text)
        if add_cls and ids[0] != CLS_TOK[1]:
            ids.insert(0, CLS_TOK[1])
        if add_bos and ids[0] != BOS_TOK[1]:
            ids.insert(0, BOS_TOK[1])
        if add_eos and ids[-1] != EOS_TOK[1]:
            ids.append(EOS_TOK[1])
        return ids

    def decode_ids(self, ids: List[int], trunc_eos=False) -> str:
        """
        convert ids to text
        :param ids:
        :param trunc_eos: skip everything after first EOS token in sequence
        :return:
        """
        if trunc_eos:
            try:
                ids = ids[:ids.index(EOS_TOK[1])]
            except ValueError:
                pass
        return super(Field, self).decode_ids(ids)

    def tokenize(self, text: str) -> List[str]:
        return self.encode_as_pieces(text.encode())

    def detokenize(self, tokens: List[str]) -> str:
        return ''.join(tokens).replace('▁', ' ').strip()

    @staticmethod
    def train(model_type: str, vocab_size: int, model_path: str, files: Iterator[str],
              no_split_toks: Optional[List[str]] = None):
        """
        Train Sentence Piece Model
        :param model_type: sentence piece model type: {unigram, BPE, word, char}
        :param vocab_size: target vocabulary size
        :param model_path: where to store model
        :param files: input files
        :param no_split_toks: Don't split these tokens
        :return:
        """
        model_prefix = model_path.replace('.model', '')
        files = set(files)  # remove duplicates
        arg = f"--input={','.join(files)} --vocab_size={vocab_size} --model_prefix={model_prefix}" \
              f" --model_type={model_type} --pad_id={PAD_TOK[1]} --bos_id={BOS_TOK[1]}" \
              f" --eos_id={EOS_TOK[1]} --unk_id={UNK_TOK[1]} --hard_vocab_limit=false"
        if no_split_toks:
            arg += f" --user_defined_symbols={','.join(no_split_toks)}"
        log.info(f"SPM: {arg}")
        SentencePieceTrainer.Train(arg)
        log.info("Training complete")
        if not model_path.endswith('.model'):
            model_path += '.model'
        return Field(model_path)


class LookupField:

    def __init__(self, vocab_file: Union[str, Path], unk_tok: str = '<unk>', pad_tok='<pad>'):
        self.int_to_str: List[str] = read_lines(vocab_file)
        self.str_to_int: Mapping[str, int] = {tok: idx for idx, tok in enumerate(self.int_to_str)}
        assert unk_tok in self.str_to_int
        assert pad_tok in self.str_to_int
        self.unk_tok: str = unk_tok
        self.unk_id: int = self.str_to_int[unk_tok]
        assert len(self.int_to_str) == len(self.str_to_int)
        self._len = len(self.int_to_str)

    def encode_as_id(self, text: str) -> int:
        return self.str_to_int.get(text, self.unk_id)

    def remap(self, text: str) -> str:
        return text if text in self.str_to_int else self.unk_tok

    def __contains__(self, item):
        return item in self.str_to_int

    def __getitem__(self, item):
        if type(item) is int:
            return self.int_to_str[item]
        elif type(item) is str:
            return self.str_to_int[item]
        else:
            raise Exception()

    def __len__(self):
        return self._len


"""
Terminology:
  - Utterance: Something the character says. Includes character's id and the text 
  - Chat: A sequence of utterances as context and the response utterance. 
         This is made up for feeding into our neural model. We control the chat context size.
  - Dialog: A bigger chat, the one that happened in the story. We don't control how big this can go.
      We make several Chat s out of Dialog
  - DialogReader: An utility for reading Dialogs from a text file
  - DialogMiniBatch: A batch of Chat s that we make to take full advantage of GPUs
     since GPUs are great for batched linear algebra ops

"""


# if eq=False, id based hashing will be used
@dataclass(eq=False)
class Utterance:
    char: Union[int, str]  # the character or speaker who said this
    text: Union[List[str], List[int]]  # the text content
    raw_text: str = None  # In case you like to keep the original text line for inspection
    raw_char: str = None  # in case you like to keep the original character name
    uid: Optional[str] = None  # In case you want to identify this by referencing to ext corpus
    weight: Optional[float] = None   # weight such as sampling weight

    def __len__(self):
        return len(self.text)

    @property
    def is_raw(self):
        assert type(self.char) == type(self.text[0]) == type(self.text[-1])  # Just a shallow check
        return type(self.char) is str


@dataclass
class ChatRecIdx:
    # index of ChatRec
    """
    The integers here are indices . You cannot interpret these without a List[Utterance]
    """
    context: List[int]
    resp: int

    def __len__(self):
        return len(self.context)


@dataclass
class ChatRec:
    context: List[Utterance]
    response: Utterance

    def as_dialog_mini_batch(self, sort_dec=True):
        """
        Convert this ChatRec to DialogMiniBatch
        :return:
        """
        chat_idx = ChatRecIdx(list(range(len(self.context))), len(self.context))
        utters = self.context + [self.response]
        raw_batch = DialogMiniBatchRaw(utters, [chat_idx])
        if sort_dec:
            raw_batch.sort_desc()
        return DialogMiniBatch(raw_batch)


@dataclass()
class Dialog:
    chat: List[Utterance] = field(default_factory=list)

    def append(self, utter: Utterance):
        self.chat.append(utter)

    def __len__(self):
        return len(self.chat)

    def __iter__(self):
        return self.chat

    def as_mini_chats(self, min_ctx: int, max_ctx: int, model_chars: Optional[Set] = None,
                      min_resp_len: int = -1, no_repeat=False,
                      down_sample=True) -> Iterator[ChatRec]:
        """

        :param min_ctx: minimum context size for the response
        :param max_ctx: maximum context size for the response
        :param model_chars: set of character whose responses you are trying to model
            if None is supplied (default), no filtering will be performed
        :param min_resp_len: responses shorter than this will be skipped. value < 1 will disable
            this filter
        :param no_repeat: do not repeat chat with smaller ctx window
        :param down_sample: sub sample most frequent responses
        :return:
        """
        assert 0 < min_ctx <= max_ctx

        for idx in range(min_ctx-1, len(self)):
            start = max(0, idx - max_ctx)
            ctx = self.chat[start: idx]  # until this idx
            resp = self.chat[idx]
            if model_chars and resp.char not in model_chars:
                continue
            if down_sample and resp.weight < 0.999:
                # if weight is 0.1 then only 10% of times this response should be chosen
                #  and given to optimizer to update with its parameters
                if resp.weight < random.uniform(0, 1):
                    continue  # Ignore when satisfied, accept when failed. Am I doing it right?

            if min_resp_len > 0 and len(resp.text) < min_resp_len:
                # ignore shorter responses
                continue
            while len(ctx) >= min_ctx:
                # shallow copy.copy is intentional,
                #  copy.deepcopy mess the id based hashing, so don't do that
                yield ChatRec(copy.copy(ctx), resp)
                if no_repeat:   # don't repeat this with smaller ctx size
                    break
                else:
                    # Slide the window, by removing the left most
                    ctx.pop(0)

    def as_test_chats(self, min_ctx: int, max_ctx: int, test_chars: Set):
        """
        :param min_ctx: minimum context window
        :param max_ctx: maximum context window
        :param test_chars: only return chats that have responses from these characters
        :return: an iterator of ChatRec
        """
        assert 0 < min_ctx <= max_ctx
        for right in range(min_ctx, len(self)):
            left = max(0, right - max_ctx)
            ctx = self.chat[left: right]        # right is not included
            resp = self.chat[right]              # right is the response
            if not test_chars or resp.char in test_chars:
                yield ChatRec(ctx, resp)


class RawDialogReader:
    """
    This one reads Raw text.
    For integer ids, see DialogReader
    """

    def __init__(self, inp: Union[str, Path, TextIO, Iterator[str]],
                 text_field: Field = None,
                 char_field: LookupField = None,
                 max_seq_len: int = 100, add_eos=True):
        """

        :param inp: dialog seq file
        :param text_field: provide this field if you want to map text to word ids.
         by default it splits words by white space and return words as sequence
        :param char_field: provide this field if you want to map character name to id.
        """
        if type(inp) is str:
            inp = Path(inp)
        if isinstance(inp, Path):
            assert inp.exists()
            inp = IO.reader(inp).open()
        self.reader = inp
        self.text_field = text_field
        self.char_field = char_field
        self.max_seq_len = max_seq_len
        self.add_eos = add_eos
        self.num_cols = 0

    def __iter__(self) -> Iterator[Dialog]:
        count = 0
        dialog = Dialog()
        for line in self.reader:
            line = line.strip()
            if not line:
                if len(dialog) > 0:
                    yield dialog
                    count += 1
                    dialog = Dialog()
                continue
            parts = line.split("\t")
            if len(parts) < self.num_cols:
                log.error(f"Skipping the line: {line}")
                continue
            self.num_cols = max(len(parts), self.num_cols)
            raw_char, raw_text = parts[-2:]
            uid = parts[0] if len(parts) > 2 else None
            char = raw_char = raw_char.strip()
            if self.char_field:
                char = self.char_field.encode_as_id(raw_char)
            if self.text_field:
                seq = self.text_field.encode_as_ids(raw_text, add_eos=True)
            else:
                seq = raw_text.strip().split()
                if self.add_eos and seq[-1] != EOS_TOK[0]:
                    seq.append(EOS_TOK[0])
            seq = seq[:self.max_seq_len]
            utter = Utterance(char, seq, raw_text=raw_text, raw_char=raw_char, uid=uid)
            dialog.append(utter)
        if len(dialog) > 0:
            count += 1
            yield dialog
        log.info(f"Read {count} dialogs")

        try:
            self.reader.close()
        except:
            pass


class DialogReader:
    """
    This one works with processed data i.e. word ids
    """

    def __init__(self, path: Path, shuffle=True, add_eos=True):
        """
        :param path: path to read TSV
        """
        assert path.exists()
        self.path = path
        self.shuffle = shuffle
        self._mem = None
        self.add_eos = add_eos

    @staticmethod
    def read_all(path: Path, add_eos):
        count = 0
        with IO.reader(path) as reader:
            dialog = Dialog()
            for line in reader:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    char, seq = parts[-2:]  # the last two are mandatory
                    uid = parts[0] if len(parts) > 2 else None
                    weight = float(parts[1]) if len(parts) > 3 else None
                    char, seq = int(char), [int(x) for x in seq.strip().split()]
                    if add_eos and seq[-1] != EOS_TOK_IDX:
                        seq.append(EOS_TOK_IDX)
                    dialog.append(Utterance(char, seq, uid=uid, weight=weight))
                else:
                    if len(dialog) > 0:
                        yield dialog
                        count += 1
                        dialog = Dialog()
            if len(dialog) > 0:
                count += 1
                yield dialog
        log.info(f"Read {count} dialogs")

    def __iter__(self):
        if self.shuffle:
            if not self._mem:
                log.info("Going to shuffle using a buffer. If this causes OOM, don't blame me!")
                self._mem = list(self.read_all(self.path, add_eos=self.add_eos))
            random.shuffle(self._mem)
            dialogs = self._mem
        else:
            dialogs = self.read_all(self.path, add_eos=self.add_eos)
        yield from dialogs


@dataclass
class DialogMiniBatchRaw:
    utters: List[Utterance] = field(default_factory=list)
    chats: List[ChatRecIdx] = field(default_factory=list)
    has_ref: bool = True

    @property
    def is_empty(self):
        return len(self.chats) == 0

    def sort_desc(self):
        """
        Sort descending order of sequence lengths.
        There is two levels of sorting involved.
            level1: sort the chats based on length of context
            level2: sort the utterances based on number of words
        :return:
        """

        # Step1: sort chats based on context length
        self.chats: List[ChatRecIdx] = sorted(self.chats, key=lambda x: len(x), reverse=True)

        # step 2: sort Utterance based on sequence length (number of words)
        #       Not so easy, we need to map the indexes in chat afters
        old_idx_uttrs = enumerate(self.utters)
        sorted_idx_uttrs = sorted(old_idx_uttrs, key=lambda idx_n_uttr: len(idx_n_uttr[1]),
                                  reverse=True)
        new_n_old_idx_uttrs = list(enumerate(sorted_idx_uttrs))
        old_to_new_idx = {new_idx: old_idx for new_idx, (old_idx, uttr) in new_n_old_idx_uttrs}

        self.utters: List[Utterance] = [uttr for _, (_, uttr) in new_n_old_idx_uttrs]
        for chat in self.chats:
            chat.context = [old_to_new_idx[old_idx] for old_idx in chat.context]
            # response will be missing for test records
            chat.resp = old_to_new_idx[chat.resp]

    @staticmethod
    def new(utters: List[Utterance], chats: List[ChatRec], sort_desc=True, pad=True):
        # going to indexify these
        utters: List[Utterance] = list(utters)  # make a copy

        utter_map = {utter: idx for idx, utter in enumerate(utters)}
        idxed_chats = [ChatRecIdx([utter_map[ut] for ut in chat.context],
                                  utter_map[chat.response])
                       for chat in chats]
        batch = DialogMiniBatchRaw(utters, idxed_chats)
        if sort_desc:
            batch.sort_desc()
        result = DialogMiniBatch(batch) if pad else batch
        return result

    def print(self):
        print(f"Dialog: {id(self):X}, Utters: {len(self.utters)}; Chats: {len(self.chats)}")
        for idx, ut in enumerate(self.utters):
            print(f"{idx} :: {ut.char}: {ut.text}")
        for idx, chat in enumerate(self.chats):
            print(f"  {idx} :: {chat.context} --> {chat.resp}")


class DialogMiniBatch:

    # Tensors
    def __init__(self, batch: DialogMiniBatchRaw, max_utter_len=80):
        self.n_utters = len(batch.utters)
        self.n_chats = len(batch.chats)
        max_utter_len = min(max_utter_len, max(len(u) for u in batch.utters))
        self.utters = torch.zeros(self.n_utters, max_utter_len, dtype=torch.long, device=device)
        self.utter_lens = torch.zeros(self.n_utters, dtype=torch.long, device=device)
        self.chars = torch.zeros(self.n_utters, dtype=torch.long, device=device)
        for i, utter in enumerate(batch.utters):
            assert utter.text[-1] == EOS_TOK_IDX
            # TODO: should I put EOS for the truncated seqs?
            seq = utter.text[:max_utter_len]
            self.utter_lens[i] = len(seq)
            self.chars[i] = utter.char
            self.utters[i, :len(seq)] = tensor(seq)

        self.chat_lens = tensor(list(map(len, batch.chats)), dtype=torch.int)
        max_chat_len = self.chat_lens.max()

        # -1 is padding, it will be incremented later
        self.chat_ctx_idx = torch.full((self.n_chats, max_chat_len), -1,
                                       dtype=torch.long, device=device)

        for i, chat in enumerate(batch.chats):
            self.chat_ctx_idx[i, :len(chat)] = tensor(chat.context, dtype=torch.int)
        self.chat_ctx_idx += 1  # zero is padding
        self.has_ref = batch.has_ref
        self.chat_resp_idx = tensor([c.resp for c in batch.chats], dtype=torch.long)
        # task: prepare response sequence
        self.resp_seqs = torch.index_select(self.utters, 0, self.chat_resp_idx)
        self.resp_chars = torch.index_select(self.chars, 0, self.chat_resp_idx)
        self.resp_lens = torch.index_select(self.utter_lens, 0, self.chat_resp_idx)
        self.max_resp_len = self.resp_lens.max()
        self.tot_resp_toks = self.resp_lens.sum()

    def to_beamed_batch(self, beam_size):

        # its short for beamed_batch (not baby)
        bb = copy.deepcopy(self)

        #  batch repeated beam_size times
        bb.n_chats = self.n_chats * beam_size
        bb.chat_ctx_idx = self.chat_ctx_idx.repeat(1, beam_size).view(bb.n_chats, -1)
        bb.chat_lens = self.chat_lens.unsqueeze(1).repeat(1, beam_size).view(bb.n_chats)
        bb.resp_chars = self.resp_chars.repeat(1, beam_size).view(bb.n_chats)

        if bb.has_ref:
            bb.chat_resp_idx = self.chat_resp_idx.unsqueeze(1).repeat(1, beam_size).view(bb.n_chats)
            bb.resp_seqs = self.resp_seqs.repeat(1, beam_size).view(bb.n_chats, -1)
            bb.resp_lens = self.resp_lens.unsqueeze(1).repeat(1, beam_size).view(bb.n_chats)
            bb.tot_resp_toks = bb.resp_lens.sum()
        return bb

    def print(self):
        print(f"Dialog: {id(self):X}, Utters: {self.n_utters}; Chats: {self.n_chats}")
        print(torch.cat([self.chars.unsqueeze(1), self.utter_lens.unsqueeze(1), self.utters, ],
                        dim=1))
        print(torch.cat([self.chat_lens.unsqueeze(1), self.chat_ctx_idx,
                         self.chat_resp_idx.unsqueeze(1)], dim=1))


class OrderedSet(dict):
    """
    Sigh, why is this not part of python standard library? (Java I miss you!)
    """

    def maybe_update(self, items):
        new_items = [it for it in items if it not in self]
        start_idx = len(self)
        update = {itm: start_idx + idx for idx, itm in enumerate(new_items)}
        self.update(update)

    def maybe_add(self, item):
        if item not in self:
            self[item] = len(self)

    def to_list(self):
        return list(self.keys())


class DialogBatchReader:

    def __init__(self, reader: DialogReader, min_ctx: int = 2, max_ctx: int = 10,
                 max_utters: int = 10, max_chats: int = 20, sort_desc: bool = True,
                 pad=True, model_chars: Optional[Set] = None, min_resp_len: int=-1,
                 no_repeat: bool=False, down_sample: bool=True):
        assert 0 < min_ctx <= max_ctx
        assert 0 < max_chats <= max_utters

        self.reader: Iterator[Dialog] = reader
        self.min_ctx = min_ctx
        self.max_ctx = max_ctx
        self.max_utters = max_utters  # Note: this is not guaranteed
        self.max_chats = max_chats
        self.sort_desc = sort_desc
        self.pad = pad
        self.model_chars = model_chars
        self.no_repeat = no_repeat
        if self.no_repeat:
            log.info("No repeat sub ctx enabled")
        if min_resp_len > 0:
            log.info(f"Ignoring responses shorter than {min_resp_len} from {reader.path}")
        self.min_resp_len = min_resp_len
        self.last_count = -1
        self.down_sample = down_sample
        if self.down_sample:
            log.info("Sub sampling enabled. This will down sample most frequent responses")

    def __iter__(self):
        count = 0
        utters: OrderedSet = OrderedSet()
        chats: List[ChatRec] = list()

        def utters_space():
            return self.max_utters - len(utters)

        def chat_space():
            return self.max_chats - len(chats)

        for dialog in self.reader:
            for chat in dialog.as_mini_chats(min_ctx=self.min_ctx, max_ctx=self.max_ctx,
                                             model_chars=self.model_chars,
                                             min_resp_len=self.min_resp_len,
                                             no_repeat=self.no_repeat,
                                             down_sample=self.down_sample):
                utters.maybe_update(chat.context)  # this might exceed max_utters, but that's okay
                utters.maybe_add(chat.response)
                chats.append(chat)
                if utters_space() <= 0 or chat_space() <= 0:
                    batch = DialogMiniBatchRaw.new(utters.to_list(), chats=chats,
                                                   sort_desc=self.sort_desc, pad=self.pad)
                    yield batch
                    count += 1
                    utters.clear()
                    chats.clear()
        if chats:  # left over in the buffer
            yield DialogMiniBatchRaw.new(utters.to_list(), chats=chats, sort_desc=self.sort_desc,
                                         pad=self.pad)
            count += 1
        if count != self.last_count:
            log.info(f"Produced {count} dialog batches")
            self.last_count = count


def _test_batching_():
    """
    A quick test on how batching works
    :return:
# Without sort
===batch:0==
Dialog: 11A353240, Utters: 5; Dialogs: 4
0 :: 0: ['u00', 'u01', 'u02', 'u03', 'u04', 'u05']
1 :: 1: ['u10', 'u11', 'u12']
2 :: 2: ['u20', 'u21', 'u22', 'u23', 'u24']
3 :: 3: ['u30', 'u31', 'u32', 'u33', 'u34', 'u35']
4 :: 4: ['u40', 'u41']
  0 :: [0, 1] --> 2
  1 :: [0, 1, 2] --> 3
  2 :: [1, 2] --> 3
  3 :: [0, 1, 2, 3] --> 4

....

===batch:3==
Dialog: 11A3531D0, Utters: 7; Dialogs: 3
0 :: 3: ['u30', 'u31', 'u32', 'u33', 'u34', 'u35']
1 :: 4: ['u40', 'u41']
2 :: 5: ['u50', 'u51', 'u52', 'u53', 'u54', 'u55']
3 :: 6: ['u60', 'u61', 'u62', 'u63', 'u64']
4 :: 10: ['b00', 'b01']
5 :: 11: ['b10', 'b11', 'b12']
6 :: 12: ['b20']
  0 :: [0, 1, 2] --> 3
  1 :: [1, 2] --> 3
  2 :: [4, 5] --> 6

## Sorted
===batch:0==
Dialog: 11E4FE358, Utters: 5; Dialogs: 4
0 :: 0: ['u00', 'u01', 'u02', 'u03', 'u04', 'u05']
1 :: 3: ['u30', 'u31', 'u32', 'u33', 'u34', 'u35']
2 :: 2: ['u20', 'u21', 'u22', 'u23', 'u24']
3 :: 1: ['u10', 'u11', 'u12']
4 :: 4: ['u40', 'u41']
  0 :: [0, 3, 2, 1] --> 4
  1 :: [0, 3, 2] --> 1
  2 :: [0, 3] --> 2
  3 :: [3, 2] --> 1

 .....

===batch:3==
Dialog: 11E4FE2E8, Utters: 7; Dialogs: 3
0 :: 3: ['u30', 'u31', 'u32', 'u33', 'u34', 'u35']
1 :: 5: ['u50', 'u51', 'u52', 'u53', 'u54', 'u55']
2 :: 6: ['u60', 'u61', 'u62', 'u63', 'u64']
3 :: 11: ['b10', 'b11', 'b12']
4 :: 4: ['u40', 'u41']
5 :: 10: ['b00', 'b01']
6 :: 12: ['b20']
  0 :: [0, 2, 3] --> 5
  1 :: [2, 3] --> 5
  2 :: [1, 4] --> 6

# Padding

Dialog: 116E7C9E8, Utters: 4; Chats: 3
   >>   [char_id, len, -- utter --- ] padding=0
tensor([[ 0,  6,  0,  1,  2,  3,  4,  5],
        [ 3,  6, 30, 31, 32, 33, 34, 35],
        [ 2,  5, 20, 21, 22, 23, 24,  0],
        [ 1,  3, 10, 11, 12,  0,  0,  0]])
  >>  [len, -- ctx --, resp] padding=-1
tensor([[ 3,  0,  3,  2,  1],
        [ 2,  0,  3, -1,  2],
        [ 2,  3,  2, -1,  1]], dtype=torch.int32)
"""
    seq = [
        Utterance(0, ['u00', 'u01', 'u02', 'u03', 'u04', 'u05']),
        Utterance(1, ['u10', 'u11', 'u12']),
        Utterance(2, ['u20', 'u21', 'u22', 'u23', 'u24']),
        Utterance(3, ['u30', 'u31', 'u32', 'u33', 'u34', 'u35']),
        Utterance(4, ['u40', 'u41']),
        Utterance(5, ['u50', 'u51', 'u52', 'u53', 'u54', 'u55']),
        Utterance(6, ['u60', 'u61', 'u62', 'u63', 'u64']),
    ]
    d1 = Dialog(chat=seq)
    d2 = Dialog(chat=[
        Utterance(10, ['b00', 'b01']),
        Utterance(11, ['b10', 'b11', 'b12']),
        Utterance(12, ['b20']),
    ])
    r: Iterator[DialogMiniBatchRaw] = DialogBatchReader(
        reader=[d1, d2], min_ctx=2, max_ctx=4, max_utters=5, max_chats=5, sort_desc=True,
        pad=False)

    for i, b in enumerate(r):
        print(f"===batch:{i}==")
        b.print()

    d3 = Dialog(chat=[
        Utterance(0, [0, 1, 2, 3, 4, 5]),
        Utterance(1, [10, 11, 12]),
        Utterance(2, [20, 21, 22, 23, 24]),
        Utterance(3, [30, 31, 32, 33, 34, 35])])

    r: Iterator[DialogMiniBatch] = DialogBatchReader(
        reader=[d3], min_ctx=2, max_ctx=4, max_utters=5, max_chats=5, sort_desc=True,
        pad=True)
    for i, b in enumerate(r):
        print(f"===batch:{i}==")
        b.print()


class LoopingIterable:
    """
    An iterable that keeps looping until a specified count is reached
    """

    def __init__(self, iterable, total: int = sys.maxsize):
        self.itr = iterable
        self.total = total
        self.count = 0

    def __iter__(self):
        self.count = 0  # reset
        looping = True
        while looping:
            for item in self.itr:
                yield item
                self.count += 1
                if self.count >= self.total:
                    looping = False
                    break


def read_tsv(path: str):
    assert os.path.exists(path)
    with IO.reader(path) as f:
        yield from (line.split('\t') for line in f)


def read_lines(path: Union[str, Path]):
    with IO.reader(path) as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        return lines


def tokenize(strs: List[str]) -> List[List[str]]:
    return [s.split() for s in strs]
