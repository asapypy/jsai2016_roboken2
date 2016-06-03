#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: jsai2016models.py
# Author: Shin Asakawa <asakawa@ieee.org>
# CreationData: 07/Mar/2016

from __future__ import print_function
import sys  # this is for debug for a while ....

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.evaluation import accuracy


"""For JSAI2016,
   Author: Shin Asakawa <asakawa@ieee.org>
   CreationData: 07/Mar/2016

   We defined several models to comare among them.
   0. naiveLM: a naive baseline model, no dropuout, one lstm
   1. RNNLM:  the original chainer ptb language model
   2. JSAI2016DIALOGUE
   3. JSAI2016DIALOGUE_NODROPOUT
"""

class naviveLM(chainer.Chain):
    """This is a naive baseline model to chech performance.
    In this model, there are two hidden layers,
    one is the embeded layer, and another is RNN layer.
    """
    def __init__(self, n_vocab, n_units, train=True):
        super(naiveLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            hidden=L.LSTM(n_units, n_units),
            output=L.Linear(n_units, n_vocab),
        )
        self.train = train

    def reset_state(self):
        self.hidden.reset_state()

    def __call__(self, x):
        """No dropout, 
        """
        embdl = self.embed(x)
        hstml = self.hidden(F.linear(h0, train=self.train))
        outpl = self.output(F.linear(h1, train=self.train))
        return outpl

class RNNLM(chainer.Chain):

    """Recurrent neural net languabe model for penn tree bank corpus.

    This is an example of deep LSTM network for infinite length input.

    """
    def __init__(self, n_vocab, n_units, train=True):
        super(RNNLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        y = self.l3(F.dropout(h2, train=self.train))
        return y


class JSAI2016DIALOGUE_CLASSIFIER(chainer.link.Chain):

    """For JSAI2016,
    Definition of a new class of a classifier, which takes
    3 arguments.
    """
    compute_accuracy = True

    def __init__(self,
                 predictor,
                 context=None,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy):
        super(JSAI2016DIALOGUE_CLASSIFIER, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.y = None
        self.loss = None
        self.accuracy = True
        if context != None:
            self.context = context

    def __call__(self, x, t, context=None):
        self.y = None
        self.loss = None
        self.accuracy = True
        self.y = self.predictor(x, context)
        self.loss = self.lossfun(self.y, t)
        if self.compute_accuracy:
            self.accuracy = accuracy.accuracy(self.y, t)
        return self.loss


class JSAI2016DIALOGUE(chainer.Chain):
    """For JSAI2016,
       Recurrent neural net languabe model for penn tree bank corpus.

       We can define 3 parallel neural network models.
       1) Normal language model
       2) Q: Questioner model
       3) A: Respondant model
       When a repsondant will get a <cntnxt> token, then 
       it will start to answer the question that the questioner would ask.
       We assumed an additional input units in the hidden1 layer to add
       the content about the question.
    """

    def __init__(self, n_vocab, n_units, context, train=True):
        super(JSAI2016DIALOGUE, self).__init__(
            embed =L.EmbedID(n_vocab,           n_units),
            layer1=L.LSTM(   n_units + n_units, n_units),
            layer2=L.LSTM(   n_units,           n_units),
            layer3=L.Linear( n_units,           n_vocab),
        )
        self.train = train

    def reset_state(self):
        self.layer1.reset_state()
        self.layer2.reset_state()

    def __call__(self, x, context):
        hidden0 = self.embed(x)

        hidden1 = self.layer1(F.activation.relu.relu(
            F.array.concat.concat((hidden0, context))))
            # context implies the LSTM cell of a Questionaire.
            # This is our dialogue model for JSAI2016
        hidden2 = self.layer2(F.dropout(hidden1, train=self.train))
        y       = self.layer3(F.dropout(hidden2, train=self.train))
        return y


class JSAI2016DIALOGUE_NODROPOUT(chainer.Chain):

    """For JSAI2016
       Recurrent neural net languabe model for penn tree bank corpus.

       We define 3 parallel neural network models.
       1) Normal language model
       2) Q: Questioner model
       3) A: Respondant model
       When a repsondant will get a <cntnxt> token, then 
       it will start to answer the question that the questioner would ask.
       We assumed an additional input units in the hidden1 layer to add the content
       about the question.
    """
    def __init__(self, n_vocab, n_units, train=True):
        super(JSAI2016DIALOGUE_NODROPOUT, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            layer1=L.LSTM(n_units, n_units),
            layer1Q=L.LSTM(n_units, n_units),
            layer1A=L.LSTM(n_units, n_units),

            layer2=L.LSTM(n_units, n_units),
            layer2Q=L.LSTM(n_units, n_units),
            layer2A=L.LSTM(n_units + n_units, n_units),

            layer3=L.Linear(n_units, n_vocab),
            layer3Q=L.Linear(n_units, n_vocab),
            layer3A=L.Linear(n_units, n_vocab),
        )
        self.train = train

    def reset_state(self):
        self.layer1.reset_state()
        self.layer1Q.reset_state()
        self.layer1A.reset_state()
        self.layer2.reset_state()
        self.layer2Q.reset_state()
        self.layer2A.reset_state()

    def __call__(self, x):
        hidden0 = self.embed(x)

        hidden1  = self.layer1( hidden0)
        hidden1Q = self.layer1Q(hidden0)
        hidden1A = F.activation.relu.relu(F.array.concat.concat((hidden0, hidden1Q)))

        hidden2  = self.layer2( hidden1 )
        hidden2Q = self.layer2Q(hidden1Q)
        hidden2A = self.layer2A(hidden1A)

        y  = self.layer3( hidden2 )
        yQ = self.layer3Q(hidden2Q)
        yA = self.layer3A(hidden2A)

        return yA


class JSAI2016DIALOGUE_FULLLSTM(chainer.Chain):

    """For JSAI2016,
       Recurrent neural net languabe model for penn tree bank corpus.

       We define 3 parallel neural network models.
       1) Normal language model
       2) Q: Questioner model
       3) A: Respondant model
       When a repsondant will get a <cntnxt> token, then 
       it will start to answer the question that the questioner would ask.
       We assumed an additional input units in the h1 layer to add the content
       about the question.
    """
    def __init__(self, n_vocab, n_units, train=True):
        super(JSAI2016DIALOGUE_NODROPOUT, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            layer1 =L.LSTM(n_units + n_units, n_units),
            layer1Q=L.LSTM(n_units + n_units, n_units),
            layer1A=L.LSTM(n_units + n_units + n_units, n_units),

            layer2 =L.LSTM(n_units + n_units, n_units),
            layer2Q=L.LSTM(n_units + n_units, n_units),
            layer2A=L.LSTM(n_units + n_units, n_units),

            layer3 =L.Linear(n_units + n_units, n_vocab),
            layer3Q=L.Linear(n_units + n_units, n_vocab),
            layer3A=L.Linear(n_units + n_units, n_vocab),
        )
        self.train = train

    def reset_state(self):
        self.layer1.reset_state()
        self.layer1Q.reset_state()
        self.layer1A.reset_state()
        self.layer2.reset_state()
        self.layer2Q.reset_state()
        self.layer2A.reset_state()

#    @static_vars(hidden1prev, hidden1Qprev, hidden1Aprev,
#                 hidden2prev, hidden2Qprev, hidden2Aprev,
#                 hidden3prev, hidden3Qprev, hidden3Aprev)
#    """These _prev variables are declared to maintain the previous
#       status for each LSTM cell.
#    """
    def __call__(self, x):

        hidden0 = self.embed(x)

        hidden1  = F.activation.linear.linear(
            F.array.concat.concat((hidden0, hidden1prev)))
        hidden1prev = hidden1

        hidden1Q = F.activation.linear.linear(
            F.array.concat.concat((hidden0, hidden1Qprev)))
        hidden1Qprev = hidden1Q

        hidden1A = F.activation.linear.linear(
            F.array.concat.concat((hidden0, hidden1Q, hidden1Aprev)))
        hidden1Aprev = hidden1A

        hidden2 = F.activation.linear.linear(
            F.array.concat.concat((hidden1, hidden2prev)))
        hidden2prev = hidden2

        hidden2Q = F.activation.linear.linear(
            F.array.concat.concat((hidden1, hidden2Qprev)))
        hidden2Qprev = hidden2Q

        hidden2A = F.activation.linear.linear(
            F.array.concat.concat((hidden1, hidden2Aprev)))
        hidden2Aprev = hidden2A

        y  = self.layer3( hidden2 )
        yQ = self.layer3Q(hidden2Q)
        yA = self.layer3A(hidden2A)

        return yA

class JSAI2016ENCDEC(chainer.Chain):
    """For JSAI2016,
       Recurrent neural net languabe model for penn tree bank corpus.

       When a repsondant will get a <cntnxt> token, then 
       it will start to answer the question that the questioner would ask.
       We assumed an additional input units in the hidden1 layer to add
       the content about the question.
    """

    def __init__(self, n_vocab, n_units, train=True):
        super(JSAI2016ENCDEC, self).__init__(
            embed =L.EmbedID(n_vocab,  n_units),
            enc1=L.LSTM(  n_units,     n_units),
            enc2=L.LSTM(  n_units,     n_units),
            enc3=L.Linear(n_units,     n_vocab),
            dec1=L.LSTM(  n_units * 2, n_units),
            dec2=L.LSTM(  n_units,     n_units),
            dec3=L.Linear(n_units,     n_vocab),
        )
        self.train = train

    def reset_state(self):
        self.enc1.reset_state()
        self.enc2.reset_state()
        self.dec1.reset_state()
        self.dec2.reset_state()

    def __call__(self, x):
        hid0 = self.embed(x)
        enc_hid1 = self.enc1(F.dropout(hid0, train=self.train))
        enc_hid2 = self.enc2(F.dropout(enc_hid1, train=self.train))
        enc_y    = self.enc3(F.dropout(enc_hid2, train=self.train))

        dec_hid1 = self.dec1(F.activation.relu.relu(
            F.array.concat.concat((hid0, enc_hid2))))
            # enc_hid2 implies the LSTM cell of the Questionaire.
            # This is our dialogue model for JSAI2016
        dec_hid2 = self.dec2(F.dropout(dec_hid1, train=self.train))
        dec_y    = self.dec3(F.dropout(dec_hid2, train=self.train))
        return enc_y, dec_y
        """ちがうなー。encoder が一回発言を終了するたびに decoder を回さねば
        ならない。だからこれでは，系列制御になっていない。
        """

