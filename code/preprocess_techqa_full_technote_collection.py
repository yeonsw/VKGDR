#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
# Example code for reading the technote corpus
import os

import argparse
import json
import jsonlines
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_path", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()
    return args

def main(args):
    inf = open(args.inp_path, 'r')
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with jsonlines.open(args.out_path, "w") as writer:
        for line in tqdm(inf):
            arr = json.loads(line)
            for entry in arr:
                if isinstance(entry, str):
                    continue
                writer.write(entry)
    return 0

if __name__ == "__main__":
    args = parse_args()
    _ = main(args)
