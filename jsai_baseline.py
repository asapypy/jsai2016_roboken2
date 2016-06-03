#!/usr/bin/env python
# --encoding: utf-8

# FileName: jsai_baseline.py
# Author: Shin Asakawa <asakawa@ieee.org>
# CreationData: 07/Mar/2016
"""For JSAI2016:

   This code was intended to give a baseline performance to compare
   the original ptb model.

The original of this file is the sample code of Chainer:
Sample script of recurrent neural network language model.

This code is ported from following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import print_function
import six
import numpy as np
import argparse
import math
import sys
import time
from datetime import datetime

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import jsai2016models

parser = argparse.ArgumentParser()
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--initmodelQ', '-q', default='',
                    help='Initialize the modelQ from given file')
parser.add_argument('--initmodelA', '-a', default='',
                    help='Initialize the modelA from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--savefile', '-s', default='jsai_baseline',
                    help='File name sufix to be saved')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=100, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=650, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=1000,
                    help='learning minibatch size')
parser.add_argument('--bproplen', '-l', type=int, default=200,
                    help='length of truncated BPTT')
parser.add_argument('--gradclip', '-c', type=int, default=1,
                    help='gradient norm threshold to clip')
parser.add_argument('--train', '-t', default='jsai_unk0.train.data',
                    help='train data file name')
parser.add_argument('--valid', '-v', default='jsai_unk0.valid.data',
                    help='valid data file name')
parser.add_argument('--test', '-x', default='jsai_unk0.test.data',
                    help='test data file name')
parser.add_argument('--lr', '-z', type=float, default=0.1,
                    help='learning ratio a hyperparameter')
parser.add_argument('--seed', '-S', type=int, default=1,
                    help='seed of random number generator')
# parser.add_argument('--test', dest='test', action='store_true')
# parser.set_defaults(test=False)

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

# Prepare dataset (preliminary download dataset by ./download.py)
vocab = {}

def load_data(filename):
    global vocab, n_vocab, rdict, words
    # words.append('PAD')
    words = []  ### this is a list
    words.append('PAD')

    ### print('### filename=%s' % (filename))
    with open(filename) as f:
        lines = f.readlines()

    line_count = 0
    for line in lines:
        for w in line.strip().split():
            words.append(w)
        # line_count += 1

    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
        # print('<i=%d, word=%s, dataset[%d]=%s>' % (i, word, i, dataset[i]))
    rdict = dict((v, k) for k, v in vocab.iteritems())
 
    # ### print('### line count=%d line has %d lines' %
    #            (line_count, len(lines)))
    # n = 0
    # line = 0
    # for word in words:
    #     n += len(word)
    #     line += 1

    # for i, word in enumerate(words[:200]):
    #     print('<i=%d,' % (i), end='')
    #     print('word=%s,' % (word), end='')
    #     print('words[i]=%s,' % (words[i]), end='')
    #     print('vocab[word]=%s,' % (vocab[word]), end='')
    #     print('dataset[i]=%s,' % (dataset[i]), end='')
    #     print('rdict[vocab[word]]=%s>' % (rdict[vocab[word]]))
    #     # (i,word,dataset[i],vocab[word],rdict[vocab[word]]))
    # print('len(words)=%s' % (len(words)))
    # print('len(dataset)=%s' % (len(words)))

    # for i, word_id in enumerate(dataset[:100]):
    #     print('<i=%d, word_id=%d' % (i, word_id), end=' ')
    #     print('rdict[%d]=%s,' % (word_id, rdict[word_id]), end=' ')
    #     # print('vocab[i],' % (vocab[i]))
    #     print('dataset[word_id]=%d' % (dataset[word_id]), end=' ')
    #     print('words[dataset[i]]=%s' % (words[dataset[i]]))
    # print('%s' % (filename))
    # sys.exit()
    return dataset


def evaluate(dataset, model):
    # Evaluation routine
    evaluator = model.copy()  # to use different state
    evaluator.predictor.reset_state()  # initialize state

    sum_log_perp = 0
    for i in six.moves.range(dataset.size - 1):
        x = chainer.Variable(xp.asarray(dataset[i:i + 1]), volatile='on')
        t = chainer.Variable(xp.asarray(dataset[i + 1:i + 2]), volatile='on')
        loss = evaluator(x, t)
        sum_log_perp += loss.data
    return math.exp(float(sum_log_perp) / (dataset.size - 1))


def main(datafiles, params):
    """Learning loop
    """
    train_data = load_data(datafiles['train'])
    valid_data = load_data(datafiles['valid'])
    test_data  = load_data(datafiles['test'])
    whole_len  = train_data.shape[0]

    print('### train_data has ', len(train_data), ' words in this corpus.')
    print('### valid_data has ', len(valid_data), ' words in this corpus.')
    print('### test_data has ' , len(test_data),  ' words in this corpus.')
    print('### vocab=%d' % (len(vocab)))
    print('### whole_len=%d' % (whole_len))

    n_epoch   = params['n_epoch']
    n_units   = params['n_units']
    batchsize = params['batchsize']
    bprop_len = params['bprop_len']
    grad_clip = params['grad_clip']
    savefile  = params['savefile']
    lr        = params['lr']
    initmodel = params['initmodel']
    resume    = params['resume']
    print('### n_epoch = %d' % (n_epoch))
    print('### n_units = %d' % (n_units))
    print('### batchsize = %d' % (batchsize))
    print('### bprop_len = %f' % (bprop_len))
    print('### grad_clip = %f' % (grad_clip))
    print('### save filename prefix = %s' % (savefile))
    print('### learning ratio = %f' % (lr))

    # Prepare RNNLM model, defined in net.py
    lm = jsai2016models.RNNLM(len(vocab), n_units)
    model = L.Classifier(lm)
    model.compute_accuracy = False   # we want the accuracy
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Setup optimizer
    optimizer = optimizers.SGD(lr=lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

    # Init/Resume
    if initmodel:
        print('Load model from', initmodel)
        serializers.load_npz(initmodel, model)
    if resume:
        print('Load optimizer state from', resume)
        serializers.load_npz(resume, optimizer)


    # Learning loop
    whole_len = train_data.shape[0]
    print('### whole_len=', whole_len)
    print('### batchsize=', batchsize, '  # length of a minibatch')

    jump = whole_len // batchsize
    print('### jump=', jump,'  # interval between minibatches')
    cur_log_perp = xp.zeros(())

    epoch = 0
    start_at   = time.time()
    cur_at     = start_at
    accum_loss = 0
    batch_idxs = list(range(batchsize))
    print('### going to train {} iterations'.format(jump * n_epoch))

    try:
        for iter in six.moves.range(jump * n_epoch):
            x = chainer.Variable(xp.asarray(
                [train_data[(jump * iter + j) % whole_len]
                 for j in batch_idxs]))
            t = chainer.Variable(xp.asarray(
                [train_data[(jump * iter + j + 1) % whole_len]
                 for j in batch_idxs]))
            loss_i = model(x, t)
            accum_loss += loss_i
            cur_log_perp += loss_i.data

            if (iter + 1) % bprop_len == 0:  # Run truncated BPTT
                model.zerograds()
                accum_loss.backward()
                accum_loss.unchain_backward()  # truncate
                accum_loss = 0
                optimizer.update()

            if (iter + 1) % jump == 0:
                now = time.time()
                throuput = float(jump) / (now - cur_at)
                perp = math.exp(float(cur_log_perp) / jump)
                print('iter %d training perplexity: %f (%f iters/sec)' %
                      (iter + 1, perp, throuput))
                cur_at = now
                cur_log_perp.fill(0)

                epoch += 1
                # print('evaluate')
                now = time.time()
                perp = evaluate(valid_data, model)
                print('iter %d validation perplexity: %f' % (iter + 1, perp))
                cur_at += time.time() - now  # skip time of evaluation

                if epoch >= 6:
		    optimizer.lr /= 1.001
		    # optimizer.lr /= 1.2
		    print('learning rate =', optimizer.lr)

            sys.stdout.flush()

    except KeyboardInterrupt:
        print("Training interupted")

    # Save the model and the optimizer
    print('save the model')
    strtime = datetime.now().strftime('%Y%m%d%H%M%S')
    serializers.save_npz('%s___%s.model' % (savefile, (strtime)), model)
    print('save the optimizer')
    serializers.save_npz('%s___%s.state' % (savefile, (strtime)), optimizer)

    # Evaluate on test dataset
    print('test')
    test_perp = evaluate(test_data, model)
    print('test perplexity:', test_perp)


if __name__ == '__main__':
    args = parser.parse_args()
    train = args.train
    valid = args.valid
    test  = args.test

    n_epoch = args.epoch        # number of epochs
    n_units = args.unit         # number of units per layer
    batchsize = args.batchsize  # minibatch size
    bprop_len = args.bproplen   # length of truncated BPTT
    grad_clip = args.gradclip   # gradient norm threshold to clip
    seed      = args.seed       # seed of random number generator
    xp.random.seed(seed)
    params = {'n_epoch':args.epoch,
              'n_units':args.unit,
              'batchsize':args.batchsize,
              'bprop_len':args.bproplen,
              'grad_clip':args.gradclip,
              'savefile':args.savefile,
              'initmodel':args.initmodel,
              'resume':args.resume,
              'lr':args.lr}

    datafiles = {'train':train, 'valid':valid, 'test':test}
    main(datafiles, params)
