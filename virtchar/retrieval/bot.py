#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 11/28/18

from virtchar.retrieval.unisenc import SentenceEncoder
import numpy as np
from virtchar.utils import IO
from typing import List
from virtchar import log
import pickle
import argparse
import traceback
from typing import Dict
import sys

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

    def __init__(self, sent_enc: SentenceEncoder, msg_reprs, resps, msgs=None):
        self.enc = sent_enc
        assert len(msg_reprs) == len(resps)
        if msgs:
            assert len(msgs) == len(resps)
        self.msg_reprs = msg_reprs
        self.resps = resps
        self.msgs = msgs
        self.enc.maybe_gpu()

    def find_closest(self, msg_rep, k):
        # TODO change it to cosine
        squared_err = np.square(self.msg_reprs - msg_rep)
        top_k = np.argsort(np.sqrt((np.sum(squared_err, axis=1))))[:k]
        return top_k

    def respond(self, query, k) -> List[str]:
        other_query = [query]
        query_embedding = self.enc.encode(other_query)
        closest_resp_idx = self.find_closest(query_embedding, k)
        return [self.resps[idx] for idx in closest_resp_idx]

    @classmethod
    def prepare(cls, msgs: List[str], resps: List[str], sent_enc_model: str):
        assert len(msgs) == len(resps)
        log.info(f"total message-responses : {len(msgs)}")
        uniq_msgs = len(set(msgs))
        if len(msgs) != uniq_msgs:
            # that's why  I am not using dictionary mapping, instead using two lists
            log.warning(f"Found Duplicate messages.  {uniq_msgs} are unique of {len(msgs)}")
        senc = SentenceEncoder(sent_enc_model)
        msg_reprs = senc.encode(msgs, verbose=True)
        return cls(senc, msg_reprs, resps, msgs)

    def save(self, path):
        log.info(f"Storing to {path}")
        # The reason for doing this crazy stuff is to increase the portability of models
        # if we simply dump object as pickle, then torch version must be matched during re-loading
        # So we dump only the state params and arrays
        with open(path, 'wb') as f:
            self.enc.to_cpu()
            state = dict(enc_state=self.enc.get_state(),
                         msg_reprs=self.msg_reprs,
                         resps=self.resps,
                         msgs=self.msgs)
            pickle.dump(state, f)
            self.enc.maybe_gpu()

    @classmethod
    def load(cls, path):
        log.info(f"Loading from  {path}")
        with open(path, 'rb') as f:
            state: Dict = pickle.load(f)
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


def chat_console(model_path, k=2, **args):
    bot: ChatBot = ChatBot.load(model_path)
    import readline
    helps = [(':quit', 'Exit'),
             (':help', 'Print this help message'),
             (':k=<int>', 'Set k to a new number (k as in KNN search)')
             ]

    def print_cmds():
        for cmd, msg in helps:
            print(f"\t{cmd:15}\t-\t{msg}")

    print("Launching Interactive shell...")
    print_cmds()
    print_state = True
    args['k'] = k
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
                k = int(line.replace(':k=', ''))
                args['k'] = k
            else:
                msg = line
                topk_resps = bot.respond(msg, k=k)
                for i, r in enumerate(topk_resps):
                    print(f">> {i:2d}: {r}")
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
