# File Name: Layer.py
# Author: Zainab Alaweel
# Date: 04 /20 / 2019
import numpy as np
from ActivationType import ActivationType

class Layer(object):
    def __init__(self,numNeurons, inputSize, activationType,batchSize, dropout = 1.0, momentumBN = 0.8):
        self.numNeurons = numNeurons;
        self.batchSize = int(batchSize);
        self.Dropout = dropout;
        self.activationType= activationType
        self.momentumBN = momentumBN
        self.inputSize=inputSize
        self.W = np.random.uniform(low=-0.1,high=0.1,size=(numNeurons,inputSize))
        self.B = np.random.uniform(low=-1,high=1,size=(numNeurons,1))
        self.GradW = np.zeros((self.batchSize, numNeurons,inputSize))
        self.GradB = np.zeros((self.batchSize, numNeurons,1))
        self. Mu = np.zeros((numNeurons,1)) 
        self.Var = np.zeros((numNeurons,1)) 
        self. Beta = np.zeros((numNeurons,1))
        self.Gamma = np.ones((numNeurons,1))
        self.dGamma = np.zeros((numNeurons,1))
        self.dBeta = np.zeros((numNeurons,1))
        self.Xhat = np.zeros((self.batchSize, numNeurons,1))
        self. Ivar = np.zeros((numNeurons,1))
        self.runningMu = np.zeros((numNeurons,1))
        self.runningVar = np.zeros((numNeurons,1))
        self.Delta = np.zeros((self.batchSize, numNeurons,1))
        self.DropM = np.zeros((self.batchSize, numNeurons,1))
        self.A = np.zeros((self.batchSize, numNeurons,1))
        self.APrime = np.zeros((self.batchSize, numNeurons,1))
        self.epsilon = 1e-8
        self.Sum = np.zeros((self.batchSize, numNeurons,1))
        #self.Sb = np.zeros((numNeurons,numNeuronsPrevLayer))
        #self.deltabn = np.zeros((numNeurons,numNeuronsPrevLayer))

    def InitializeDropoutMatrix(self, DM ):
         if (self.Dropout < 1.0):
             for i in range (DM.shape[0]):
                for j in range (DM.shape[1]):
                    num = np.random.rand(1)
                    if (num < self.Dropout):
                            DM[i,j] = 1/self.Dropout;
                    else:
                        DM[i, j] = 0;

    def Evaluate (self,inputData, bi, useBatchNorm, doBNTestMode = False):
        #print("input too layer: ", inputData)
        #print("input shape: ", inputData.shape, " weight shape: ", self.W.shape)
        #print("weights: ", self.W)

        self.Sum[bi] = np.dot(self.W, inputData) + self.B
        #print("Sume: ", self.Sum)
        if ((useBatchNorm == True) & (doBNTestMode == False)):
            x_minus_mu = self.Sum[bi] - self.Mu;
            self.Ivar = 1.0/np.sqrt(self.Var + self.epsilon)
            self.Xhat[bi] = np.multiply(x_minus_mu, self.Ivar)
            self.Sum[bi] = self.Xhat[bi] * self.Gamma + self.Beta

        if ((useBatchNorm == True) & (doBNTestMode == True)):
            x_minus_Runningmu = self.Sum[bi] - self.runningMu
            for kk in range (self.runningVar.shape[0]):
                self.Xhat[bi] = x_minus_Runningmu/ np.sqrt(self.runningVar + self.epsilon)
                self.Sum[bi] = self.Xhat[bi] * self.Gamma + self.Beta
        # Apply Drop out 
        if (self.Dropout < 1.0):
            self.InitializeDropoutMatrix(self.DropM[bi])
            self.Sum[bi] = self.Sum[bi] * self.DropM[bi]

        if (self.activationType == ActivationType.SIGMOID):
            self.A[bi] = self.sigmoid(self.Sum[bi])
            self.APrime[bi] = self.A[bi] * ( 1 - self.A[bi])  # A * (1 - A)

        if (self.activationType == ActivationType.RELU):
            self.A[bi] = self.Relu(self.Sum[bi])  #no aprime for relu - delta is computed accordingly during training

        if (self.activationType == ActivationType.SOFTMAX):
           self.A[bi] =  self.Softmax(self.Sum)
        #print("X-M: ", x_minus_Runningmu)
        #print("IVAR: " ,  self.Ivar)
        #print("Xhat: ", self.Xhat)
        #print("Sum shape: ",self.Sum.shape)
        #print("Sum: " , self.Sum)
        #print("A: ", self.A)

        return self.A[bi];

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x)) 
    def TanH(self, x):
        return np.tanh(x)
    def Relu(self, x):
        return np.maximum(0,x)
  
    def Softmax(self, x):
        if (x.shape[0] == x.size):
            ex = np.exp(x)
            return ex/ex.sum()
        ex = np.exp(x)
        for i in range(ex.shape[0]):
            denom = ex[i,:].sum()
            ex[i,:] = ex[i,:]/denom
        return ex