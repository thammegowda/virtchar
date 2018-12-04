#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 11/28/18

from virtchar.retrieval.unisenc import SentenceEncoder
from virtchar.utils import IO
from typing import List
from virtchar import log, device
import argparse
import traceback
from typing import Dict, Tuple
import sys
import torch

"""
## Terminology: Message - Response model
Message is what other person sends
Response is what our chat bot responds
Sentence is either message or a response

## Abbreviations
msg : message
resp : response
repr: representation
sent: sentence
when you add 's' at the end, its a plural --> its a list like collection 
"""


class ChatBot:
    """
    Inspired by https://github.com/Spandan-Madan/Me_Bot/blob/master/Me_Bot.ipynb
    """

    def __init__(self, sent_enc: SentenceEncoder, msg_reprs, resp_reprs, resps, msgs=None):
        self.enc = sent_enc
        assert len(msg_reprs) == len(resps)
        if msgs:
            assert len(msgs) == len(resps)
        self.msg_reprs: torch.Tensor = msg_reprs
        self.resp_reprs: torch.Tensor = resp_reprs
        self.resps = resps
        self.msgs = msgs
        self.maybe_gpu()

    def maybe_gpu(self):
        self.enc.maybe_gpu()
        self.msg_reprs = self.msg_reprs.to(device)
        self.resp_reprs = self.resp_reprs.to(device)

    @staticmethod
    def find_closest(qry_repr, doc_reprs, metric, k: int):
        # TODO change it to cosine
        # euclidean distance or L2
        if metric == 'L2':
            dists = (doc_reprs - qry_repr).pow(2).sum(dim=-1).sqrt()
            sorted_dists, indices = dists.sort()
            return [(sorted_dists[i], indices[i]) for i in range(k)]
        elif metric == 'cosine':
            sims = torch.nn.functional.cosine_similarity(doc_reprs, qry_repr)
            sorted_sims, indices = sims.sort(descending=True)
            return [(sorted_sims[i], indices[i]) for i in range(k)]
        else:
            raise Exception(f'{metric} not implemented')

    def _find_closest(self, query: str, doc_repr, metric: str = 'L2', k: int = 4):
        queries = [query]
        qry_reprs = self.enc.encode(queries)
        qry_reprs = torch.from_numpy(qry_reprs).to(device)
        return self.find_closest(qry_reprs, doc_repr, metric=metric, k=k)

    def message_to_response(self, msg, metric, k) -> List[Tuple[float, str]]:
        """
        Input is other character's utterance. Response is this character's response
        :param msg:
        :param k:
        :return:
        """
        closest_resp_idx = self._find_closest(msg, self.msg_reprs, metric=metric, k=k)
        return [(dist, self.resps[idx]) for dist, idx in closest_resp_idx]

    def response_to_response(self, resp, metric, k) -> List[Tuple[float, str]]:
        """
        Input
        :param resp:
        :param k:
        :return:
        """
        closest_resp_idx = self._find_closest(resp, self.resp_reprs, metric=metric, k=k)
        return [(dist, self.resps[idx]) for dist, idx in closest_resp_idx]

    @classmethod
    def prepare(cls, msgs: List[str], resps: List[str], sent_enc_model: str):
        assert len(msgs) == len(resps)
        log.info(f"total message-responses : {len(msgs)}")
        uniq_msgs = len(set(msgs))
        if len(msgs) != uniq_msgs:
            # that's why  I am not using dictionary mapping, instead using two lists
            log.warning(f"Found Duplicate messages.  {uniq_msgs} are unique of {len(msgs)}")
        senc = SentenceEncoder(sent_enc_model)
        log.info("Encoding messages")
        msg_reprs = senc.encode(msgs, verbose=True)
        log.info("Encoding responses")
        resp_reprs = senc.encode(resps, verbose=True)
        return cls(senc, torch.from_numpy(msg_reprs), torch.from_numpy(resp_reprs), resps, msgs)

    def save(self, path):
        log.info(f"Storing to {path}")
        # The reason for doing this crazy stuff is to increase the portability of models
        # if we simply dump object as pickle, then torch version must be matched during re-loading
        # So we dump only the state params and arrays
        state = dict(enc_state=self.enc.get_state(),
                     msg_reprs=self.msg_reprs,
                     resp_reprs=self.resp_reprs,
                     resps=self.resps,
                     msgs=self.msgs)
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        log.info(f"Loading from  {path}")
        state: Dict = torch.load(path, map_location=torch.device('cpu'))
        enc = SentenceEncoder(state_dict=state.pop('enc_state'))
        obj: ChatBot = cls(sent_enc=enc, **state)
        return obj


def read_msg_resp(path: str):
    def _read(rdr):
        recs = (x.strip() for x in rdr)
        recs = (x for x in recs if x)
        recs = (x.split('\t') for x in recs)
        recs = (x for x in recs if len(x) == 2)
        recs = list(recs)
        msgs = [x[0] for x in recs]
        resps = [x[1] for x in recs]
        return msgs, resps

    if type(path) is str:
        with IO.reader(path) as r:
            return _read(r)
    else:
        return _read(path)


def prepare(enc_path: str, chats, model_path):
    msgs, resps = read_msg_resp(chats)
    bot = ChatBot.prepare(msgs, resps, sent_enc_model=enc_path)
    bot.save(model_path)


def chat_console(model_path, k=2, metric='L2', imitate=False, **args):
    bot: ChatBot = ChatBot.load(model_path)
    import readline
    helps = [(':quit', 'Exit'),
             (':help', 'Print this help message'),
             (':k=<int>', 'Set k to a new number (k as in KNN search)'),
             (':imitate', 'Flip the imitate flag. When imitate=True, bot tries to imitate you,'
                          '(default: bot tries to respond instead of imitation)'),
             (':metric=<L2 | cosine>', 'Set metric type')
             ]

    def print_cmds():
        for cmd, msg in helps:
            print(f"\t{cmd:15}\t-\t{msg}")

    print("Launching Interactive shell...")
    print_cmds()
    print_state = True
    args.update(dict(k=k, metric=metric, imitate=imitate))
    while True:
        if print_state:
            state = '  '.join(f'{k}={v}' for k, v in args.items())
            print('\t|' + state)
        print_state = False
        line = input('<<: ')
        line = line.strip()
        if not line:
            continue
        try:
            if line == ':quit':
                break
            elif line == ':help':
                print_cmds()
            elif line.startswith(':k='):
                args['k'] = k = int(line.replace(':k=', ''))
                print_state = True
            elif line.startswith(':metric='):
                line = line.replace(':metric=', '').strip()
                assert line in ('cosine', 'L2')
                args['metric'] = metric = line
                print_state = True
            elif line.startswith(':imitate'):
                # Flip the state
                args['imitate'] = imitate = not imitate
                print_state = True
            else:
                if imitate:
                    topk_resps = bot.response_to_response(line, metric=metric, k=k)
                else:
                    topk_resps = bot.message_to_response(line, metric=metric, k=k)
                for i, (score, resp) in enumerate(topk_resps):
                    print(f">> {i:2d}: {score:g} : {resp}")
        except EOFError as e1:
            break
        except Exception as e2:
            traceback.print_exc()
            print_state = True


def main():
    par = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub_par = par.add_subparsers(dest='cmd')
    sub_par.required = True

    prep_par = sub_par.add_parser('prep')
    prep_par.add_argument('-ep', '--enc-path', required=True,
                          help='path to SentenceEncoder state')
    prep_par.add_argument('-mp', '--model-path', required=True,
                          help='path to store Bot model')
    prep_par.add_argument('-c', '--chats', help='Training chats: Message \\t Response',
                          default=sys.stdin)

    chat_par = sub_par.add_parser('chat')
    chat_par.add_argument('-mp', '--model-path', required=True,
                          help='Model state adapted from "prep"')
    args = vars(par.parse_args())
    {
        'chat': chat_console,
        'prep': prepare
    }[args.pop('cmd')](**args)


if __name__ == '__main__':
    main()
