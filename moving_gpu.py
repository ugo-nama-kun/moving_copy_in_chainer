# -*- coding: utf-8 -*-
"""
@author: naoto-y
"""

import time
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(1, 20),
            l2=L.Linear(20, 1),
             )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        y = F.sigmoid(self.l2(h1))
        return y
    
    
class Model(Chain):
    def __init__(self, predictor):
        super(Model, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        self.loss = F.sigmoid_cross_entropy(y, t)
        return self.loss, y
    

model = Model(MLP()) # original network
model_moving = copy.deepcopy(model) # network for copy

# GPU-nization
model.to_gpu()
model_moving.to_gpu()

# Optimizer for original network
optimizer = optimizers.AdaGrad(lr=0.1)
optimizer.setup(model)


# Training setting
N = 3000
tau = 0.001
n_test = 100 # # of test set

# variables for plotting
loss_hist = np.nan * np.zeros(N)
loss_hist_moving = np.nan * np.zeros(N)

plt.figure(1)
for i in xrange(N):
    # TEST for original network
    x_ = np.array(np.random.rand(n_test,1), dtype=np.float32)
    x = Variable(cuda.to_gpu(x_))
    t = Variable(cuda.to_gpu((x_ > 0.5).astype(np.int32)))
    
    loss, pred = model(x, t)
    loss_hist[i] = loss.data
    
    # TEST for moving-copy network    
    model.zerograds()
    loss_moving, pred = model_moving(x, t)
    #print "LOSS:::" ,loss_moving.data
    loss_hist_moving[i] = loss_moving.data

    
    ## Training by single sample
    # Simple input
    x = Variable(cuda.to_gpu(np.array([[np.random.rand()]], dtype=np.float32)))
    # Simple target
    if x.data > 0.5:
        t = Variable(cuda.to_gpu(np.array([[1]], dtype=np.int32)))
    else:
        t = Variable(cuda.to_gpu(np.array([[0]], dtype=np.int32)))
    
    model.zerograds()
    loss, pred = model(x, t)
    loss.backward()
    optimizer.update()
    print x.data, " , ", t.data, " , ", pred.data
    
    
    # Moving copy
    model_moving.predictor.l1.W.data = tau * model.predictor.l1.W.data + (1-tau) * model_moving.predictor.l1.W.data 
    model_moving.predictor.l1.b.data = tau * model.predictor.l1.b.data + (1-tau) * model_moving.predictor.l1.b.data
    model_moving.predictor.l2.W.data = tau * model.predictor.l2.W.data + (1-tau) * model_moving.predictor.l2.W.data
    model_moving.predictor.l2.b.data = tau * model.predictor.l2.b.data + (1-tau) * model_moving.predictor.l2.b.data
    
    # loss plotting
    if np.mod(i, 100)==0:
        plt.clf()
        plt.plot(loss_hist, "r")
        plt.hold(True)
        plt.plot(loss_hist_moving, "b")
        plt.draw()
        time.sleep(0.001)
        plt.pause(0.0001)
        plt.hold(False)
    
    
plt.clf()
plt.plot(loss_hist, "r")
plt.hold(True)
plt.plot(loss_hist_moving, "b")
plt.legend(["loss_original_network","loss_moving_copy"])
plt.draw()
time.sleep(0.001)
plt.pause(0.0001)
plt.hold(False)
