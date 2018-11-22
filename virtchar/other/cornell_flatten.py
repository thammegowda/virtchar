#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 11/21/18


import argparse
import sys
import logging as log
from dataclasses import dataclass

log.basicConfig(level=log.INFO)


def parse_line(line, delim="+++$+++"):
    line = line.replace("\t", " ").strip()
    return [col.strip() for col in line.split(delim)]


def parse_chat(line):
    rec = parse_line(line)
    rec[-1] = eval(rec[-1])
    return rec


def read_movie_lines(inp):
    recs = map(parse_line, inp)
    return {rec[0]: MovieLine(*rec) for rec in recs}


@dataclass
class MovieLine:
    line_id: str
    char_id: str
    movie_id: str
    char_name: str
    text: str


def main(lines, chat, out):
    utters_map = read_movie_lines(lines)
    chats = map(parse_chat, chat)
    for c in chats:
        convs = [utters_map[l] for l in c[-1]]
        for l in convs:
            out.write(f'{l.line_id}\t{l.char_name}\t{l.text}\n')
        out.write("\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-l', '--lines', type=argparse.FileType('r', encoding='utf-8', errors='ignore'),
                   default=sys.stdin,
                   help='Input file path of movie lines')
    p.add_argument('-c', '--chat', type=argparse.FileType('r', encoding='utf-8', errors='ignore'),
                   default=sys.stdin,
                   help='Input file path of movie conversations')
    p.add_argument('-o', '--out', type=argparse.FileType('w', encoding='utf-8', errors='ignore'),
                   default=sys.stdout,
                   help='Output file path')
    args = vars(p.parse_args())
    main(**args)
