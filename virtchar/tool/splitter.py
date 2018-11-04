#!/usr/bin/env python
#
# Author: Thamme Gowda [tg at isi dot edu] 
# Created: 10/22/18
import argparse
import yaml
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--splits", dest="split_map", type=Path, required=True,
                   help="season ids to split names mapping in a yaml file")
    p.add_argument("-in", "--inp", help="Input TSV file containing all data",
                   type=Path, required=True)
    p.add_argument("-o", "--out", help="Output Directory path",
                   type=Path, required=True)
    args = p.parse_args()
    return vars(args)


def main(split_map: Path, inp: Path, out: Path):
    assert split_map.exists()
    assert inp.exists()
    out.mkdir(exist_ok=True)
    splits = ['train', 'dev', 'test']
    out_files = {sp: open(out / f'{sp}.tsv', 'w', encoding='utf-8', errors='ignore')
                 for sp in splits}
    with split_map.open() as f:
        mapping = yaml.load(f)
        assert all(x in mapping for x in splits)
    out_map = {s: out_files[sp] for sp, ss in mapping.items() for s in ss}

    with inp.open() as inp:
        for rec in inp:
            season = rec.split("\t")[0].split("_")[0]
            out_file = out_map[season]
            out_file.write(rec)


if __name__ == '__main__':
    args = parse_args()
    main(**args)
