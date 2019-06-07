# File Name: CNNLayer.py
# Author: Zainab Alaweel
# Date: 04 /20 / 2019
from ActivationType import ActivationType
from PoolingType import PoolingType
from FeatureMap import *
import numpy as np
from scipy import signal

class CNNLayer(object):
    def __init__(self, numFeatureMaps, numPrevLayerFeatureMaps, inputSize, kernelSize, poolingType, activationType, batchSize):
        self.batchSize = batchSize
        self.numFeatureMaps = numFeatureMaps
        self.numPrevLayerFeatureMaps = numPrevLayerFeatureMaps
        self.batchSize = inputSize
        self.kernelSize = kernelSize
        self.FeatureMapList = [] # creating list of feature maps in each layer
        self.convOutputSize = inputSize - kernelSize + 1;
        self.ConvolResults = np.zeros((batchSize, numPrevLayerFeatureMaps , numFeatureMaps, self.convOutputSize, self.convOutputSize ))
        self.Kernels = np.zeros(( numPrevLayerFeatureMaps , numFeatureMaps, kernelSize, kernelSize))
        self.KernelGrads =  np.zeros((numPrevLayerFeatureMaps , numFeatureMaps, kernelSize, kernelSize))
        self.ConvSums = np.zeros((batchSize, numFeatureMaps, self.convOutputSize,self.convOutputSize))
        for i in range (numFeatureMaps):
            fpm = FeatureMap(self.convOutputSize, activationType, poolingType, batchSize)
            self.FeatureMapList.append(fpm)

        # Initialize kernels 
        for i in range (numPrevLayerFeatureMaps):
            for j in range (numFeatureMaps):
                self.Kernels[i,j]=np.random.uniform(low=-0.1,high=0.1,size=(kernelSize, kernelSize))
        #print("Kernels \n" , self.Kernels)
        # Evaluate each layer 
    def Evaluate(self, PrevLayerOutputList, batchIndex):
        #print("PrevLayerOutputList ", PrevLayerOutputList)
        # Do Convolution between the output from pevious layer and the kernels
        for p in range (0, self.numPrevLayerFeatureMaps):
            for q in range (0, self.numFeatureMaps):
                self.ConvolResults[batchIndex,p,q] = signal.convolve2d(PrevLayerOutputList[p], self.Kernels[p,q], mode = 'valid')
        #print("ConvolResults[batchIndex,p,q]" ,  self.ConvolResults[batchIndex])

        # Add Convolution Results 
        for q in range (len(self.FeatureMapList)):
            self.ConvSums[batchIndex, q] = np.zeros(( self.convOutputSize,self.convOutputSize))
            for p in range (len(PrevLayerOutputList)):

                self.ConvSums[batchIndex, q] = self.ConvSums[batchIndex, q] + self.ConvolResults[batchIndex, p,q]
        #print("ConvSums[batchIndex] \n " , self.ConvSums[batchIndex])
        # Evaluate each feature map 
        for i in range (len(self.FeatureMapList)):
           (self.FeatureMapList[i].Evaluate(self.ConvSums[batchIndex, i], batchIndex))

            