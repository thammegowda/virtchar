#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 11/28/18

# Universal Sentence Encoder


from models import InferSent
import torch
from typing import List, Union
import argparse
import numpy as np
import sys
import logging as log
from virtchar.utils import IO

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
log.info(f"Device = {device_name}")
device = torch.device(device_name)
cpu_device = torch.device('cpu')


VERSION = 2
MODEL_CONF = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
              'pool_type': 'max', 'dpout_model': 0.0, 'version': VERSION}


class SentenceEncoder:
    """
    Universal sentence encoder, based on https://github.com/facebookresearch/InferSent
    """

    def __init__(self, state_path=None, state_dict=None):
        assert bool(state_path) != bool(state_dict), 'Either state_path or state_dict must be there'
        self.model = InferSent(config=MODEL_CONF)
        if state_path:
            log.info(f"Loading state from {state_path}")
            state = torch.load(state_path, map_location=device)
        else:
            state = state_dict
        assert 'model' in state and 'word_vec' in state  # created by self.prepare() method

        self.model.load_state_dict(state['model'])
        self.model.word_vec = state['word_vec']
        self.maybe_gpu()

    def encode(self, sentences, tokenize=True, **kwargs):
        return self.model.encode(sentences, tokenize=tokenize, **kwargs)

    def maybe_gpu(self, device=device):
        self.model = self.model.to(device)

    def to_cpu(self):
        self.model = self.model.to(cpu_device)

    @staticmethod
    def prepare(model_path: str, word_vecs: str, out_path: str,
                sentences: Union[str, List[str]] = None,
                max_vocab: int = 0):
        """
        this method is for adapting the vocabulary,
        :param model_path: unadapted model state
        :param word_vecs: word vectors
        :param out_path: where to store the state
        :param sentences: training sentences for scanning the vocabulary
        :param max_vocab: maximum vocabulary size (optional)
        :return:
        """
        assert bool(sentences) != bool(max_vocab), 'Either sentences or max_vocab should be given'

        model = InferSent(config=MODEL_CONF)
        log.info(f"Loading state from {out_path}")

        model.load_state_dict(torch.load(model_path))
        log.info(f"Loading word vecs from {out_path}")
        model.set_w2v_path(word_vecs)
        if sentences:
            if type(sentences) is not list:
                sentences = list(read_lines(sentences))
            log.info("Building vocabulary from sentences")
            model.build_vocab(sentences, tokenize=True)
        if max_vocab:
            log.info(f"Pruning vocabulary to top {max_vocab} types")
            model.build_vocab_k_words(K=max_vocab)
        log.info(f"Saving at {out_path}")

        state = SentenceEncoder._get_state(model)
        torch.save(state, out_path)

    @classmethod
    def _get_state(cls, model):
        if isinstance(model, cls):
            model = model.model
        # by default InferSent doesnt pickle word_vec, so this hack
        return {'model': model.state_dict(), 'word_vec': model.word_vec}

    def get_state(self):
        return self._get_state(self)


def read_lines_reader(reader):
    for line in reader:
        line = line.strip()
        if line:
            yield line


def read_lines(path):
    if type(path) is str:
        with IO.reader(path) as reader:
            yield from read_lines_reader(reader)
    else:
        return read_lines_reader(path)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def print_row(source, sentences, row):
    print(f">> {source}")
    for i, sent in enumerate(sentences):
        print(f'{i}\t{row[i]:.4f}\t{sentences[i]}')


def main(model_path, inp):
    senc = SentenceEncoder(state_path=model_path)
    n = 10
    lines = list(read_lines(inp))[:n]
    n = min(n, len(lines))
    sent_repr = senc.encode(lines)
    row = [0.0] * n

    for i in range(n):
        for j in range(n):
            row[j] = cosine(sent_repr[i], sent_repr[j])
        print_row(lines[i], lines, row)


if __name__ == '__main__':
    par = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub_par = par.add_subparsers(dest='cmd')
    sub_par.required = True

    prep_par = sub_par.add_parser('prep')
    prep_par.add_argument('-i', '-mp', '--model-path', required=True,
                          help='Model state for adapting (download from'
                               ' https://s3.amazonaws.com/senteval/infersent/infersent2.pkl)')
    prep_par.add_argument('-v', '-wv', '--word-vecs', required=True,
                          help='Vectors (download and extract from'
                               ' https://s3-us-west-1.amazonaws.com/fasttext-vectors/'
                               'crawl-300d-2M.vec.zip)')
    prep_par.add_argument('-t', '-s', '--train', dest='sentences', help='Training sentences')
    prep_par.add_argument('-mv', '--max-vocab', help='Maximum Vocabulary Size', type=int)
    prep_par.add_argument('-o', '--out-path', help="Where to store the model state after adapting",
                          required=True)

    enc_par = sub_par.add_parser('enc')
    enc_par.add_argument('-mp', '--model-path', required=True,
                         help='Model state adapted from "prep"')
    enc_par.add_argument('-i', '--inp', default=sys.stdin,
                         help='Computer cosine sim between these sentences')
    args = vars(par.parse_args())
    {
        'enc': main,
        'prep': SentenceEncoder.prepare
    }[args.pop('cmd')](**args)
