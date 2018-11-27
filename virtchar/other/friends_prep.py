#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 11/26/18


import argparse
import sys
import logging as log

log.basicConfig(level=log.INFO)


def read_dialogs(inp):
    dialog = []
    for line in inp:
        line = line.strip()
        if not line:     # Empty line
            if dialog:
                yield dialog
                dialog = []
            continue

        parts = line.split('\t')
        if len(parts) != 3:
            log.warning(f"Skipping bad line: {line}")
        elif parts[1] in {'<event>', '<scene>'}:
            if dialog:
                yield dialog
                dialog = []
        else:
            dialog.append(parts)


def write_dialogs(dialogs, out, sep='\n'):
    stats = {'dialogs': 0,
             'utters': 0,
             'max_dialog_len': 0,
             'max_utter_len': 0}
    for dialog in dialogs:
        for rec in dialog:
            line = '\t'.join(rec) + '\n'
            out.write(line)
            stats['utters'] += 1
            stats['max_utter_len'] = max(stats['max_utter_len'], len(rec[-1]))
        stats['dialogs'] += 1
        stats['max_dialog_len'] = max(stats['max_dialog_len'], len(dialog))
        out.write(sep)
    log.info(stats)


def main(inp, out):
    dialogs = read_dialogs(inp)
    write_dialogs(dialogs, out=out, sep='\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path')
    args = vars(p.parse_args())
    main(**args)
