#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 10/29/18

""" NOTE: following the notation from
'A Neural Network Approach to Context-Sensitive Generation of Conversational Responses'
by Alessandro Sordoni et al (2015),
https://www.aclweb.org/anthology/N/N15/N15-1020.pdf

This script filters transcript sequence into
 - Context
 - Message or stimulus
 - Response
"""

import argparse
import sys
from virtchar import log


def filter_triples(inp, character, bod='<bod>'):
    """
    Filters dialog triples from a stream of transcripts
    :param inp: stream of transcripts
    :param character: the character we are interested in
    :param bod: Begin of Dialog token string
    :return:
    """
    # Note: currently only one message context is used
    #    TODO: extend it to longer context

    ctx = (bod, bod, bod)
    msg = (bod, bod, bod)
    for line in inp:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            log.error(f"Unable to parse line: {line}")
            continue
        id, speaker, text = parts
        should_skip = speaker == '<error>'
        if should_skip:
            continue
        should_reset = speaker == '<scene>'
        if should_reset:
            ctx = (bod, bod, bod)
            msg = (bod, bod, bod)
            continue
        if speaker == '<event>':
            text = '( ' + text + ' )'
        rsp = (id, speaker, text)
        if speaker == character:
            yield ctx, msg, rsp
        ctx, msg = msg, rsp


def write_out(triples, out):
    """
    this func just writes triple records as TSV records
    :param triples: iterator to read triples
    :param out: file stream to write output
    :return:
    """
    count = 0
    for c, m, r in triples:
        rec = list(c) + list(m) + list(r)
        out.write("\t".join(rec) + "\n")
        count += 1
    log.info(f"Wrote {count} recs to {out.name}")


def main(inp, out, character):
    triples = filter_triples(inp, character)
    write_out(triples, out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('character', type=str,
                   help='Character\'s dialog you are interested in training.')
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path; having ID \\t Character \\t Dialog')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path to write output')

    args = vars(p.parse_args())
    main(**args)
