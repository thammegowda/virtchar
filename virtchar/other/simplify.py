#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 11/20/18


import argparse
import sys
import logging as log
import unicodedata as ud
import re

log.basicConfig(level=log.INFO)

EXCLUSIONS = {'.', ',', '?'}


def is_not_punct(tok, exclusions=EXCLUSIONS):
    if tok in exclusions:
        return True
    for ch in tok:
        if not ud.category(ch).startswith('P'):
            return True
    return False


def clean(inp, no_case=False, no_puncts=False, no_brackets=False, max_len=100):
    count = 0
    for line in inp:
        count += 1
        line = line.strip()
        if not line:
            yield ''
        else:
            parts = line.split('\t')
            if len(parts) != 3:
                log.warning(f'Skipping the bad record at line {count}:: {line}')
                continue
            rec_id, char_name, text = parts
            if no_case:
                text = text.lower()
                char_name = char_name.lower()
            if no_brackets:
                text = re.sub(r'\([^)]*\)', '', text)
                text = re.sub(r'\[[^]]*\]', '', text)
                text = ' '.join(text.split())
            if no_puncts:
                toks = [tok for tok in text.split() if is_not_punct(tok)]
                text = ' '.join(toks)
            text = ' '.join(text.split()[:max_len])
            yield f'{rec_id}\t{char_name}\t{text}'


def main(inp, out, **kwargs):
    cleaned_recs = clean(inp, **kwargs)
    count = 0
    for line in cleaned_recs:
        out.write(line)
        out.write('\n')
        count += 1
    log.info(f'Wrote {count} recs')


def add_off_arg(parser, name, msg):
    dest = 'no_' + name.replace("-", "_")
    # grp = parser.add_mutually_exclusive_group()
    # grp.add_argument(f"--{name}", dest=dest, action='store_true', help=msg, default=default)
    parser.add_argument(f"--no-{name}", dest=dest, action='store_true', help=msg,
                        default=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path')
    add_off_arg(p, 'case', msg="Casing of text")
    add_off_arg(p, 'puncts', msg="Punctuations in text.")
    add_off_arg(p, 'brackets', msg="Brackets and the text inside the brackets.")
    p.add_argument('-m', '--max-len', type=int, default=60,
                   help='Maximum number of words in sequence (truncate longer seqs)')

    args = vars(p.parse_args())
    main(**args)
