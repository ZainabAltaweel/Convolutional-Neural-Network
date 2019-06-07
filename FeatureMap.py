# File Name: FeatureMap.py
# Author: Zainab Alaweel
# Date: 04 /20 / 2019
import numpy as np
from ActivationType import ActivationType
from PoolingType import PoolingType 

class FeatureMap(object):
    def __init__(self, inputDataSize, activationType, poolingType, batchSize):
        # Initializing feature map
        self.inputDataSize = inputDataSize
        self.poolingType = poolingType
        self.activationType= activationType
        self.batchSize= batchSize
        self.Sum = np.zeros((batchSize, inputDataSize, inputDataSize))
        self.ActCV = np.zeros((batchSize, inputDataSize, inputDataSize))
        self.OutputSS = np.zeros((batchSize, inputDataSize//2 , inputDataSize//2 ))
        self.DeltaSS = np.zeros((batchSize, inputDataSize//2 , inputDataSize//2 ))
        self.DeltaCV = np.zeros((batchSize, inputDataSize, inputDataSize))
        self.APrime = np.zeros((batchSize, inputDataSize, inputDataSize))
        self.Bias = 0
        #self.Bias = np.random.uniform(low=-1,high=1)
        self.BiasGrad = 0
    # Evaluation feature map
    def Evaluate(self, inputData, batchIndex):
        #print("Inputdata: ", inputData)
        #print("Bias" , self.Bias)
        # Finding Sum (adding Bias to the input data)

        self.Sum[batchIndex] = inputData + self.Bias
        # Finding ActCV (using Activation Functions)
        if (self.activationType == ActivationType.SIGMOID):
            self.ActCV[batchIndex]=  1 / (1 + np.exp(-self.Sum[batchIndex])) 
            self.APrime[batchIndex] = self.ActCV[batchIndex] * (1 - self.ActCV[batchIndex])

        if (self.activationType == ActivationType.TANH):
            self.ActCV[batchIndex]= np.tanh(self.Sum[batchIndex])
            self.APrime[batchIndex] = (1 - self.ActCV[batchIndex] *self.ActCV[batchIndex])

        if (self.activationType == ActivationType.RELU):
            self.ActCV[batchIndex] =  np.maximum(0,self.Sum[batchIndex])

        # Finding Output of feature map ( applying pooling) 
        if (self.poolingType == PoolingType.AVGPooling):
            self.OutputSS[batchIndex] = self.averagePooling(self.ActCV[batchIndex])

        if (self.poolingType == PoolingType.MAXPooling):
            self.OutputSS[batchIndex] = self.MaxPooling(self.ActCV[batchIndex])

        #print("sum of  ", batchIndex , self.Sum[batchIndex])
        #print("Sum.Shape = ", batchIndex, self.Sum.shape)
        #print("ActCv \n " ,batchIndex,  self.ActCV[batchIndex])
        #print("Out SS \n ", batchIndex , self.OutputSS[batchIndex])
        return self.OutputSS

    def averagePooling(self, x):
        rows = x.shape[0]
        cols = x.shape[0]
        AVG = np.zeros((rows//2, cols//2))
        for i in range (0 , int(rows/2)):
            for j in range (0, int(cols/2)):
                  AVG[i,j] = (x[i * 2][j * 2] + x[i * 2][j * 2 + 1] + x[i * 2 + 1][j * 2] + x[i * 2 + 1][j * 2 + 1]) / 4.0
        return AVG

    def MaxPooling(self, x):
        rows = x.shape[0]
        cols = x.shape[0]
        MAX = np.zeros((rows//2, cols//2))
        for i in range (0 , int(rows/2)):
            for j in range (0, int(cols/2)):
                  MAX[i,j] = max(x[i * 2][j * 2] , x[i * 2][j * 2 + 1] , x[i * 2 + 1][j * 2] , x[i * 2 + 1][j * 2 + 1])
        return MAX