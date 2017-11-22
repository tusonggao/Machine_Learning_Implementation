import numpy as np
from numpy import mat, shape, std, mean
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
    
    def init_OutputWB(self):
        self.out_w = 2.0*(random.rand(self.nOut, self.nHidden) - 0.5)
        self.out_b = 2.0*(random.rand(self.nOut, 1) - 0.5)
        self.out_wb = mat(self.addcol(mat(self.out_w), mat(self.out_b)))
    
    def loadDataSet(self, filename):
        self.dataMat = []; self.classLabels = []
        fr = open(filename)
        for line in fr.readlines():
            lineArr = line.strip().split()
            self.dataMat.append([float(lineArr[0]), float(lineArr[1]), 1.0])
            self.classLabels.append(int(lineArr[2]))
        self.dataMat = mat(self.dataMat)
        m, n = shape(self.datMat)
        self.nSampleNum = m
        self.nSampleDim = n-1
    
    def normalize(self, dataMat):
        [m, n] = shape(dataMat)
        for i in range(n-1):
            dataMat[:, i] = (dataMat[:, i] - mean(dataMat[:, i]))/(std(dataMat[:, i]) + 1.0e-10)
        return dataMat
            
            
            
            
            
            
            
            
            
            
            
            
            
            


if __name__=='__main__':
    print('hello world')