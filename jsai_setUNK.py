#!/usr/bin/env python
# -*- coding: utf-8 -*-

# FileName: jsai_setUNK.py
# Author: Shin Asakawa <asakawa@ieee.org>
# CreationDate: 25/Mar/2016
# I, Shin Asakawa, have all rights reserved.

from __future__ import print_function
from __future__ import division
import argparse
import six
import sys
# import nltk
from gensim import corpora, models, similarities
from collections import defaultdict
# from pprint import pprint
# import matplotlib as plot

datafilename = 'all_mlq20160522.txt'
unk_thres_default = 5
parser = argparse.ArgumentParser()
parser.add_argument('--datafile', '-f', default=datafilename,
                    help='data file name to be processed')
parser.add_argument('--unk', '-u', default=unk_thres_default,
                    help='UNK threshold to be set')

def read_datafile(filename=datafilename):
    with open(filename) as f:
        allbuf = f.readlines()
    return allbuf

def main(datafile='all_mlq20160522.txt', unk_thres=5):
    with open(datafile) as f:
        docs = [line.lower().split() for line in f.readlines()]

    freqdict = defaultdict(int)
    for line in docs:
        for word in line:
            freqdict[word] += 1

    for line in docs:
        for word in line:
            if freqdict[word] <= unk_thres:
                word = '<unk>'
            print(word, end=' ')
        print()

if __name__ == '__main__':
    args = parser.parse_args()
    datafile = args.datafile
    unk_thres = int(args.unk)
    main(datafile, unk_thres)
