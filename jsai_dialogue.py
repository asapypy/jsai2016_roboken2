#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

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
parser.add_argument('--savefile', '-s', default='jsai_dialogue',
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

# For preparing dataset
vocab = {}  ### this is a dict

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



def evaluate(dataset, model, modelQ, modelA):
    # Evaluation routine
    evaluator = model.copy()            # to use different state
    evaluator.predictor.reset_state()   # initialize state

    evaluatorQ = modelQ.copy()          # to use different state
    evaluatorQ.predictor.reset_state()  # initialize state

    evaluatorA = modelA.copy()          # to use different state
    evaluatorA.predictor.reset_state()  # initialize state

    x  = chainer.Variable(xp.asarray(dataset), volatile='on')
    t  = chainer.Variable(xp.asarray(dataset), volatile='on')
    xQ = chainer.Variable(xp.asarray(dataset), volatile='on')
    tQ = chainer.Variable(xp.asarray(dataset), volatile='on')
    xA = chainer.Variable(xp.asarray(dataset), volatile='on')
    tA = chainer.Variable(xp.asarray(dataset), volatile='on')

    x2  = chainer.Variable(xp.ndarray((1), dtype='int32'), volatile='on')
    t2  = chainer.Variable(xp.ndarray((1), dtype='int32'), volatile='on')
    xQ2 = chainer.Variable(xp.ndarray((1), dtype='int32'), volatile='on')
    tQ2 = chainer.Variable(xp.ndarray((1), dtype='int32'), volatile='on')
    xA2 = chainer.Variable(xp.ndarray((1), dtype='int32'), volatile='on')
    tA2 = chainer.Variable(xp.ndarray((1), dtype='int32'), volatile='on')

    cnt_n = vocab['<eoq>']
    pad_n = vocab['PAD']
    eos_n = vocab['<eoa>']
    sum_log_perp = 0
    xA.data[0] = pad_n
    tA.data[0] = pad_n

    q_flg = True
    pos = 0
    for j in xrange(len(x.data)):
        # if j % 1000 == 0 :
        #     print('### j=%d' % (j))
        if q_flg == True:
            xA.data[j] = pad_n
            tA.data[j] = pad_n
        else:
            xQ.data[j] = pad_n
            tQ.data[j] = pad_n
        if x.data[j] == cnt_n or x.data[j] == eos_n:
            q_flg = not q_flg
        if q_flg == False:
            xQ.data[pos + 1:] = pad_n
            tQ.data[pos + 1:] = pad_n
        else:
            xA.data[pos + 1:] = pad_n
            tA.data[pos + 1:] = pad_n

        x2.data[0]  =  x.data[j]
        t2.data[0]  =  t.data[j]
        xQ2.data[0] = xQ.data[j]
        tQ2.data[0] = tQ.data[j]
        xA2.data[0] = xA.data[j]
        tA2.data[0] = tA.data[j]
        loss = evaluator(x2, t2)
        loss = evaluatorQ(xQ2, tQ2)
        # we need adjust the size
        loss = evaluatorA(xA2, tA2, evaluatorQ.predictor.l2.c)
        sum_log_perp += loss.data
    return math.exp(float(sum_log_perp) / (dataset.size - 1))


def main(datafiles, params):
    """Learning loop
    """
    train_data = load_data(datafiles['train'])
    valid_data = load_data(datafiles['valid'])
    test_data  = load_data(datafiles['test'])
    whole_len = train_data.shape[0]

    print('### train_data has ', len(train_data), ' words in this corpus.')
    print('### valid_data has ', len(valid_data), ' words in this corpus.')
    print('### test_data has ' , len(test_data),  ' words in this corpus.')
    print('### vocab=%d' % (len(vocab)))
    print('### whole_len=%d' % (whole_len))

    n_epoch    = params['n_epoch']
    n_units    = params['n_units']
    batchsize  = params['batchsize']
    bprop_len  = params['bprop_len']
    grad_clip  = params['grad_clip']
    savefile   = params['savefile']
    lr         = params['lr']
    initmodel  = params['initmodel']
    initmodelQ = params['initmodelQ']
    initmodelA = params['initmodelA']
    resume     = params['resume']
    print('### n_epoch = %d' % (n_epoch))
    print('### n_units = %d' % (n_units))
    print('### batchsize = %d' % (batchsize))
    print('### bprop_len = %f' % (bprop_len))
    print('### grad_clip = %f' % (grad_clip))
    print('### save filename prefix = %s' % (savefile))
    print('### learning ratio = %f' % (lr))

    # Prepare RNNLM model, defined in jsai_net.py
    lm = jsai2016models.RNNLM(len(vocab), n_units)
    model = L.Classifier(lm)
    model.compute_accuracy = False

    lmQ = jsai2016models.RNNLM(len(vocab), n_units)
    modelQ = L.Classifier(lmQ)
    modelQ.compute_accuracy = False

    lmA = jsai2016models.JSAI2016DIALOGUE(len(vocab), n_units, modelQ)
    print('### lmA.__class__=', lmA.__class__)
    # modelA = L.Classifier(lmA)
    modelA = jsai2016models.JSAI2016DIALOGUE_CLASSIFIER(lmA)
    print('### modelA.__class__=', modelA.__class__)
    modelA.compute_accuracy = False

    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
        modelQ.to_gpu()
        modelA.to_gpu()

    # Setup optimizer
    # optimizer = optimizers.SGD(lr=1.)
    optimizer = optimizers.SGD(lr=lr)
    optimizer.setup(model)
    optimizer.setup(modelQ)
    optimizer.setup(modelA)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))


    # Init/Resume
    if initmodel:
        print('Load model from', initmodel)
        serializers.load_npz(initmodel, model)
    if initmodelQ:
        print('Load modelQ from', initmodelQ)
        serializers.load_npz(initmodelQ, modelQ)
    if initmodelA:
        print('Load modelA from', initmodelA)
        serializers.load_npz(initmodelA, modelA)
    if resume:
        print('Load optimizer state from', resume)
        serializers.load_npz(resume, optimizer)

    # print(train_data.shape)
    jump = 5  # Thanks for Kenji Iwai to debug
    jump = whole_len // batchsize
    ## jump indicates the number of batches to be computed,
    ## and means epochs as well. Commented by Shin asakawa.
    if whole_len < batchsize:
        jump = 1
        batchsize = whole_len
        print('### !!!! batchsize modified to %d' % (batchsize))

    print('### batchsize=', batchsize, '  # length of minibatch')
    print('### jump=', jump, '  # number of batches to be repeated, and iterations by epoch, as well')
    cur_log_perp = xp.zeros(())

    epoch      = 0
    start_at   = time.time()
    cur_at     = start_at
    accum_loss = 0
    batch_idxs = list(range(batchsize))
    print('### going to train {} iterations'.format(jump * n_epoch))

    cnt_n = vocab['<eoq>']
    pad_n = vocab['PAD']
    eos_n = vocab['<eoa>']

    try:
        for iter in six.moves.range(jump * n_epoch):
            x = chainer.Variable(xp.asarray(
                [train_data[(jump * iter + j) % whole_len]
                 for j in batch_idxs]))
            t = chainer.Variable(xp.asarray(
                [train_data[(jump * iter + j + 1) % whole_len]
                 for j in batch_idxs]))
            xQ = chainer.Variable(xp.asarray(
                [train_data[(jump * iter + j) % whole_len]
                 for j in batch_idxs]))
            tQ = chainer.Variable(xp.asarray(
                [train_data[(jump * iter + j + 1) % whole_len]
                 for j in batch_idxs]))
            xA = chainer.Variable(xp.asarray(
                [train_data[(jump * iter + j) % whole_len]
                 for j in batch_idxs]))
            tA = chainer.Variable(xp.asarray(
                [train_data[(jump * iter + j + 1) % whole_len]
                 for j in batch_idxs]))

            # for c in batch_idxs:
            #     print('t.data[%d]=%d' % (c, t.data[c]), end=' ')
            #     print('words[%d]=%s=' % (t.data[c], words[t.data[c]]), end='')
            #     print('<%d> ' % (vocab[words[t.data[c]]]), end=' ')
            #     print('%s' % (rdict[vocab[words[t.data[c]]]]))
            # sys.exit()
            xA.data[0] = pad_n
            tA.data[0] = pad_n
            pos = 0
            q_flg = True
            for j in xrange(len(x.data)):
                if q_flg == True:
                    xA.data[j] = pad_n
                    tA.data[j] = pad_n
                else:
                    xQ.data[j] = pad_n
                    tQ.data[j] = pad_n
                    if x.data[j] == cnt_n or x.data[j] == eos_n:
                        q_flg = not q_flg

            loss_i = model(x, t)
            # accum_loss += loss_i
            loss_i = modelQ(xQ, tQ)
            # accum_loss += loss_i

            # lmA.l2.c  has batchsize X number of neurons states
            loss_i = modelA(xA, tA, lmQ.l2.c)  # add the contents of modelQ
            accum_loss += loss_i
            cur_log_perp += loss_i.data

            if (iter + 1) % bprop_len == 0:  # Run truncated BPTT
                model.zerograds()
                modelQ.zerograds()
                modelA.zerograds()
                accum_loss.backward()
                accum_loss.unchain_backward()  # truncate
                accum_loss = 0
                optimizer.update()
                # optimizerQ.update()
                # optimizerA.update()

            if (iter + 1) % jump == 0:
                now = time.time()
                throuput = float(jump) / (now - cur_at)
                perp = math.exp(float(cur_log_perp) / jump)
                print('iter %d trained perplexity:%f (%f iters/sec)' %
                      (iter + 1, perp, throuput))
                cur_at = now
                cur_log_perp.fill(0)

                epoch += 1
                # print('### evaluating valid dataset')
                now = time.time()
                perp = evaluate(valid_data, model, modelQ, modelA)
                print('iter %d validation perplexity: %f' %
                      (iter + 1, perp))
                cur_at += time.time() - now  # skip time of evaluation
                if epoch >= 6:
                    optimizer.lr /= 1.001
                    # optimizer.lr  /= 1.2
                    # optimizerQ.lr /= 1.2
                    # optimizerA.lr /= 1.2
                    print('learning rate =', optimizer.lr)

            sys.stdout.flush()

    except KeyboardInterrupt:
        print("Training interupted")

    # Evaluate on test dataset
    # print('test')
    test_perp = evaluate(test_data, model, modelQ, modelA)
    print('### test perplexity:', test_perp)

    # Save the model and the optimizer
    print('### save the model')
    # serializers.save_npz('rnnlm.model', model)
    strtime = datetime.now().strftime('%Y%m%d%H%M%S')
    serializers.save_npz('%s___%s.model' %
                         (savefile, (strtime)), model)
    serializers.save_npz('%s_Q_%s.model' %
                         (savefile, (strtime)), modelQ)
    serializers.save_npz('%s_A_%s.model' %
                         (savefile, (strtime)), modelA)
    print('save the optimizer')
    serializers.save_npz('%s___%s.state' %
                         (savefile, (strtime)), optimizer)


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
              'initmodelQ':args.initmodelQ,
              'initmodelA':args.initmodelA,
              'resume':args.resume,
              'lr':args.lr}

    datafiles = {'train':train, 'valid':valid, 'test':test}
    main(datafiles, params)
