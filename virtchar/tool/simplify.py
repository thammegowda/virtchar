#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 11/20/18


import argparse
import sys
import logging as log
import unicodedata as ud

log.basicConfig(level=log.INFO)


def is_not_punct(tok):
    for ch in tok:
        if not ud.category(ch).startswith('P'):
            return True
    return False


def clean(inp, lowercase, remove_puncts):
    count = 0
    for line in inp:
        count += 1
        line = line.strip()
        if not line:
            yield ''
        else:
            parts = line.split('\t')
            assert len(parts) == 3, f'ERROR with line at {count}:: {line}'
            rec_id, char_name, text = parts
            if lowercase:
                text = text.lower()
                char_name = char_name.lower()
            if remove_puncts:
                toks = [tok for tok in text.split() if is_not_punct(tok)]
                text = ' '.join(toks)
            yield f'{rec_id}\t{char_name}\t{text}'


def main(inp, out, lowercase=True, remove_puncts=True):
    cleaned_recs = clean(inp, lowercase=lowercase, remove_puncts=remove_puncts)
    count = 0
    for line in cleaned_recs:
        out.write(line)
        out.write('\n')
        count += 1
    log.info(f'Wrote {count} recs')


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path')
    args = vars(p.parse_args())
    main(**args)


