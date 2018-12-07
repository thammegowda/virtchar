#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 12/6/18


import argparse
import sys
from typing import Iterator, List, Dict
import re
import string
from pathlib import Path
import hashlib
from collections import defaultdict
from virtchar.tool.dataprep import RawDialogReader, Dialog

punct_regex = re.compile('[%s]' % re.escape(string.punctuation))


def hash_text(text: str) -> int:
    text = text.lower().replace(' ', '')
    text = punct_regex.sub('', text)
    hex_sig = hashlib.md5(text.encode()).hexdigest()
    return int(hex_sig, 16)


def sampling_weights(clusters: List[List[str]]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for clstr in clusters:
        weight = 1.0 / len(clstr)
        for _id in clstr:
            weights[_id] = weight
    return weights


def cluster(reader: Iterator[Dialog]) -> Dict[int, List[str]]:
    clusters = defaultdict(list)
    for d in reader:
        d: Dialog = d
        for utter in d.chat:
            assert utter.uid
            assert utter.raw_text
            clusters[hash_text(utter.raw_text)].append(utter.uid)
    return clusters


def main(inp_path, out):
    dialogs = RawDialogReader(inp_path)
    clusters = cluster(dialogs)
    weights = sampling_weights(clusters.values())

    for _id, weight in weights.items():
        args.out.write(f'{_id}\t{weight}\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--inp', type=Path, help='Input file path', required=True)
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path')
    args = p.parse_args()
    main(args.inp, args.out)

