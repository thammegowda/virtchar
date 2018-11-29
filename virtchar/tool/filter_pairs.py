#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 11/28/18

"""

This script filters transcript sequence into
 - Context
 - Message or stimulus
 - Response
"""

import argparse
import sys
from virtchar import log


def filter_pairs(inp, character):
    """
    Filters dialog triples from a stream of transcripts
    :param inp: stream of transcripts
    :param character: the character we are interested in
    :return:
    """
    msg = None
    for line in inp:
        line = line.strip()
        parts = line.split("\t")
        if len(parts) < 2:
            if line:    # not an empty line
                log.error(f"Unable to parse line: {line}")
            msg = None
            continue
        speaker, resp = parts[-2:]
        should_skip = speaker == '<error>'
        if should_skip:
            continue

        if speaker == character:
            if msg:
                yield msg, resp
        msg = resp


def write_out(pairs, out):
    """
    this func  writes pairs as TSV records
    :param pairs: iterator to read pairs
    :param out: file stream to write output
    :return:
    """
    count = 0
    for rec in pairs:
        out.write("\t".join(rec) + "\n")
        count += 1
    log.info(f"Wrote {count} recs to {out.name}")


def main(inp, out, character):
    triples = filter_pairs(inp, character)
    write_out(triples, out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('character', type=str,
                   help='name of character who\'s dialog you are interested in training.')
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path; Character \\t Utterance per line.'
                        ' A blank line must separate two dialogs')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path to write output')

    args = vars(p.parse_args())
    main(**args)
