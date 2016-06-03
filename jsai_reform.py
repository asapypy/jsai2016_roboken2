#!/usr/bin/env python
# -*- coding: utf-8 -*-

# FileName: jsai_reform.py
# Author: Shin Asakawa <asakawa@ieee.org>
# CreationDate: 31/May/2016
# I, Shin Asakawa, have all rights reserved.

from __future__ import print_function
from __future__ import division
import argparse
import re
import six
import sys
import nltk

datafilename = 'all_mlq20160522.txt'
parser = argparse.ArgumentParser()
parser.add_argument('--datafile', '-f', default=datafilename,
                    help='data file name to be processed')


def read_datafile(filename=datafilename):
    with open(filename) as f:
        allbuf = f.readlines()
    return allbuf


def print_ongoing_version(allbuf):
    allstr = list()
    tmpstr = list()
    qstr = list()
    astr = list()
    q_flg = True
    a_flg = False
    for line in allbuf:
        if 'Question=' in line:
            q_flg = True
            a_flg = False
            qstr = list()  # initialize qstr
            # qstr.append('<SOS>')
            # qstr.append('<SOQ>')
            qstr.append(nltk.word_tokenize(
                line.decode('utf-8').strip().lower()))
            astr = list()  # initialize qstr
            continue

        if 'comment_no.=' in line:
            if q_flg:
                qstr.append('<EOQ>')
            q_flg = False
            # a_flg = True

            if not a_flg:
                for w in qstr:
                    allstr.append(w)
                a_flg = True
                # allstr.append('<SOA>')
                allstr.append(nltk.word_tokenize(
                    line.decode('utf-8').strip().lower()))
            else:
                allstr.append('<EOA>')
                # allstr.append('<EOS>')
                for w in qstr:
                    allstr.append(w)
                # allstr.append('<SOA>')
                allstr.append(nltk.word_tokenize(
                    line.decode('utf-8').strip().lower()))
                astr = list()
            continue

        if '-------------' in line:
            if a_flg:
                astr.append('<EOA>')
                # astr.append('<EOS>')
                for w in astr:
                    allstr.append(w)
            qstr = list()
            astr = list()  # clear answer 
            q_flg = False
            a_flg = False
            continue

        if q_flg:
            qstr.append(nltk.word_tokenize(
                line.decode('utf-8').strip().lower()))
        else:
            astr.append(nltk.word_tokenize(
                line.decode('utf-8').replace('Question=','').replace('comment no.= ','').replace('-->','').strip().lower()))

            # tmpstr.append(nltk.word_tokenize(
            #   line.decode('utf-8').strip().lower()))
            # allstr.append(nltk.word_tokenize(
            #    line.decode('utf-8').strip().lower()))
    return allstr

 
def print_ok_versoin(allbuf):
    allstr = list()
    q_flg = True
    for line in allbuf:
        if 'Question=' in line:
            allstr.append('<SOS>')
            allstr.append('<SOQ>')
            q_flg = True
        elif 'comment_no.=' in line:
            if q_flg:
                allstr.append('<EOQ>')
                allstr.append('<SOA>')
                q_flg = False
            else:
                allstr.append('<EOA>')
                allstr.append('<SOA>')
        if '-------------' in line:
            if q_flg:
                allstr.append('<EOQ>')
                allstr.append('<EOS>')
                continue
            # else:
            allstr.append('<EOA>')
            allstr.append('<EOS>')
            # q_flg = True
        else:
            allstr.append(nltk.word_tokenize(line.decode('utf-8').strip().lower()))
            # allstr.append(nltk.word_tokenize(line.decode('utf-8').replace('Question=','').replace('comment no.= ','').replace('-->','').strip().lower()))
            # # allstr = allstr[1:]

    return allstr


def print_ongoing_version_save(allbuf):
    allstr = list()
    tmpstr = list()
    q_flg = True
    for line in allbuf:
        if 'Question=' in line:
            tmpstr.append('<SOS>')
            tmpstr.append('<SOQ>')
            q_flg = True
        elif 'comment_no.=' in line:
            if q_flg:
                tmpstr.append('<EOQ>')
                q_flg = False
            else:
                tmpstr.append('<EOA>')
            tmpstr.append('<SOA>')
        if '-------------' in line:
            if q_flg:
                tmpstr = list()
            else:
                tmpstr.append('<EOA>')
                tmpstr.append('<EOS>')
            for w in tmpstr:
                allstr.append(w)
            tmpstr = list()
            q_flg = True
        else:
            tmpstr.append(nltk.word_tokenize(line.decode('utf-8').strip().lower()))
            # allstr.append(nltk.word_tokenize(line.decode('utf-8').strip().lower()))
            # allstr.append(nltk.word_tokenize(line.decode('utf-8').replace('Question=','').replace('comment no.= ','').replace('-->','').strip().lower()))
    return allstr

def allstr2freq(allstr):
    textdata = nltk.Text(allstr)  # textnize allstr into 
    textlist = []
    for line in textdata:
        if isinstance(line, str):  ## for deleting tokens as <SOA>,<EOA>,<SOQ>,<EOQ>
            textlist.append(line)
        else:
            for word in line:
                textlist.append(word.encode('utf-8'))
    freq_dist = nltk.probability.FreqDist(textlist)
    vocab = freq_dist.keys()
    return textdata, textlist, freq_dist

def print_old_version(allbuff):
    # allstr = list(allbuff)
    tokens = list()
    # line_n = 0
    q_flg = True
    for line in allbuf:
        if 'Question=' in line:
            tokens.append(nltk.word_tokenize('<SOQ>'))
            q_flg = True
        if 'comment no.=' in line:
            if q_flg:
                tokens.append(nltk.word_tokenize('<EOQ>'))
                tokens.append(nltk.word_tokenize('<SOA>'))
                q_flg = False
            else:
                tokens.append(nltk.word_tokenize('<EOA>'))
                tokens.append(nltk.word_tokenize('<SOA>'))
        if '-------------' in line:
            q_flg = True
            tokens.append(nltk.word_tokenize('<EOA>'))
        else:
            # allstr.append(line.lower())
            tokens.append(nltk.word_tokenize(line.decode('utf-8').replace('Question=','').replace('comment no.= ','').replace('-->','').strip().lower()))
        return tokens

def print_older_version(allstr):
    # tokens = nltk.word_tokenize(allstr.decode('utf-8'))
    # tokens = nltk.word_tokenize(allstr.decode('utf-8').strip().lower())
    tokens = nltk.word_tokenize(allstr.decode('utf-8').replace('-------------','').replace('Question=','').replace('comment no.= ','').replace('-->','').strip().lower())
    # tokens = nltk.word_tokenize(allstr.decode('utf-8').replace('question=','').replace('comment no.= ','').replace('-->','').strip().lower())
    print(type(tokens))
    # sys.exit()
    print(type(allstr))
    # sys.exit()
    tokens = nltk.word_tokenize(allstr)
    text = nltk.Text(tokens)
    text = allstr
    freq_dist = nltk.probability.FreqDist(text)
    print(type(text))
    print(type(vocab))
    freq_dist.plot(50)


def add_unk(text, textlist, freq_dist, threshold_num=0):
    mapping = nltk.defaultdict(lambda: 'UNK')
    for v in textlist:
        if freq_dist[v] > threshold_num:
            mapping[v] = v
    text_with_unk = [mapping[v] for v in textlist]
    text_unk_set = set(text_with_unk)
    return text_with_unk

def print_final_output(text_with_unk):
    for tok in text_with_unk:
        print(tok, end=' ')
        if '<EOA>' in tok:
            print()

def main(datafile='all_mlq20160522.txt'):
    allbuf = read_datafile(datafile)
    allstr = print_ongoing_version(allbuf)
    text, textlist, freq_dist = allstr2freq(allstr)
    print_final_output(textlist)

if __name__ == '__main__':
    args = parser.parse_args()
    # if args.datafile:
        # print('### data file name to be processed: ', args.datafile)
    datafile = args.datafile
    main(datafile)
