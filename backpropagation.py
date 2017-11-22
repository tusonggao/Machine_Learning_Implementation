import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

class BPNet(object):
    def __init__(self):
        self.eb = 0.01
        self.iterator = 0
        self.eta = 0.1
        self.mc = 0.3
        self.maxiter = 2000
        self.nHidden = 4
        self.nOut = 1
        self.errlist = []
        self.dataMat = 0
        self.classLabels = 0
        self.nSampleNum = 0
        self.nSampleDim = 0
    
    def init_hiddenWB(self):
        self.hi_w = 2.0*(random.rand(self.nHidden, self.nSampleDim)-0.5)
        self.hi_b = 2.0*(random.rand(self.nHidden, 1)-0.5)
        self.hi_wb = mat(self.addcol(mat(self.hi_w), mat(self.hi_b)))


if __name__=='__main__':
    print('hello world')