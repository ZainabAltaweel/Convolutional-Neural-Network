# File Name: DeepCNN.py
# Author: Zainab Alaweel
# Date: 04 /20 / 2019
import numpy as np
from scipy import signal
from sklearn.utils import shuffle
from ActivationType import *
from PoolingType import *
from FeatureMap import *
from CNNLayer import *
from Layer import *
class DeepCNN(object):
    """assembles the entire deep CNN network"""
    def __init__(self, CNNLayerList, layerList, inputDataList, outputLabels, batchSize):
        self.CNNLayerList = CNNLayerList
        self.LayerList = layerList
        self.InputDataList = inputDataList
        self.OutputLabels = outputLabels
        self.batchSize = batchSize
        self.Flatten = np.zeros(batchSize)

    def Evaluate(self, inputData, batchIndex):
        for i in range (0, len(self.CNNLayerList)):
            PrevLayerOutputList = []
            if (i == 0):
                PrevLayerOutputList.append(inputData)
            else:
                PrevLayerOutputList.clear()
                for j in range (len(self.CNNLayerList[i - 1].FeatureMapList)):
                    PrevLayerOutputList.append(self.CNNLayerList[i-1].FeatureMapList[j].OutputSS[batchIndex])
            #print("Previous layer output list",PrevLayerOutputList)
            self.CNNLayerList[i].Evaluate(PrevLayerOutputList,batchIndex)
        #for j in range (len(self.CNNLayerList[i].FeatureMapList)):
            #print(" Last Layer output to be flattened ",self.CNNLayerList[i].FeatureMapList[j].OutputSS)

        #flatten each feature map in the CNN layer and assemble all maps into an nx1 vector
        LastLayerOrder = len(self.CNNLayerList) - 1
        outputSSsize = self.CNNLayerList[LastLayerOrder].FeatureMapList[0].OutputSS.shape[1]
        flattenSize= outputSSsize * outputSSsize * (len(self.CNNLayerList[LastLayerOrder].FeatureMapList))
        self.Flatten=np.zeros((self.batchSize, flattenSize, 1))
        index=0
        for j in range (len(self.CNNLayerList[LastLayerOrder].FeatureMapList)):
            ss = (self.CNNLayerList[LastLayerOrder].FeatureMapList[j].OutputSS[batchIndex]).flatten()
            for k in range (ss.shape[0]):
                self.Flatten[batchIndex, index, 0]= ss[k]
                index = index + 1
        #print("Flatten shape", self.Flatten.shape)
        #print("Flatten: ",self.Flatten)
        for i in range (0, len(self.LayerList)):
            if (i==0):
                res = self.LayerList[i].Evaluate(self.Flatten[batchIndex],batchIndex,False)  # first layer
            else:
                res = self.LayerList[i].Evaluate(res,batchIndex,False)
        #print("Resrult: " , res)
        return res

    def Train(self, numEpochs, learningRate, batchSize):
        trainingError = 0
        for i in range (0, numEpochs):
            trainingError = 0
            #for sh in range(len(self.InputDataList)):
            self.InputDataList , self.OutputLabels = shuffle(self.InputDataList , self.OutputLabels, random_state=0) # is this shuffle enough or I have to shuffel the list object??
            dj=0 # input index 
            for j in range (0, len(self.InputDataList)// batchSize):
                dj = j * batchSize
                trainingErr = np.zeros(batchSize)
                for b in range (0, batchSize):
                    # Forward Pass
                    res = self.Evaluate(self.InputDataList[dj+b], b)
                    # Compute training error
                    trainingErr[b] += ((res - self.OutputLabels[dj +b]) * (res - self.OutputLabels[dj +b])).sum()
                    #print("batch index: ", b)
                    #print("training result = " , res)
                    #print("real output= ", self.OutputLabels[dj+b])
                    #print("Training error= ", trainingErr[b])
                    #print("training shape: ", trainingErr.shape)
                     
                    #compute deltas on regular NN layers
                    for count in range (len(self.LayerList)-1 , -1, -1):
                        layer = self.LayerList[count]
                        if (count == (len(self.LayerList) - 1)):  # if last layer
                            layer.Delta[b] = -self.OutputLabels[dj + b] + layer.A[b] # for softmax by default
                            if (layer.activationType == ActivationType.SIGMOID):
                                layer.Delta[b] = layer.Delta[b] * layer.APrime[b]
                            if (layer.activationType == ActivationType.RELU):
                                for m in range (0 , layer.numNeurons):
                                    if (layer.Sum[b,m,0] < 0):
                                        layer.Delta[b] = 0
                        else:  # previous layer
                            layer.Delta[b] = np.dot(self.LayerList[count + 1].W.T , self.LayerList[count + 1].Delta[b])
                            #apply dropout
                            if (layer.Dropout < 1.0):
                                layer.Delta[b] = layer.Delta[b] * (layer.DropM[b])

                            if (layer.activationType == ActivationType.SIGMOID):
                                layer.Delta[b] = layer.Delta[b] * layer.APrime[b]
                            if (layer.activationType == ActivationType.RELU):
                                for m in range (0 , layer.numNeurons):
                                    if (layer.Sum[b,m,0] < 0):
                                        layer.Delta[b, m, 0] = 0
                        #print("delta: ", count , layer.Delta)
                        layer.GradB[b] = layer.GradB[b] + layer.Delta[b]
                        if (count == 0):  # first NN layer connected to CNN last layer via Flatten
                            layer.GradW[b] = layer.GradW[b] + (layer.Delta[b] * self.Flatten[b].T) # flatten = previous output
                        else:
                            layer.GradW[b] = layer.GradW[b] + (layer.Delta[b] * self.LayerList[count - 1].A[b].T)
                        
                    #compute delta on the output of SS (flat) layer of all feature maps
                    deltaSSFlat = np.dot(self.LayerList[0].W.T , self.LayerList[0].Delta[b])
                    #print("DeltaSSFlat: " , deltaSSFlat)
                    #do reverse flattening and distribute the deltas on
                    #each feature map's SS (SubSampling layer)
                    index = 0
                    #last CNN layer
                    LayerInd = len(self.CNNLayerList) - 1 #Last CNN layer index
                    for d in range (len(self.CNNLayerList[LayerInd].FeatureMapList)):
                        self.CNNLayerList[LayerInd].FeatureMapList[d].DeltaSS[b] =np.zeros((self.CNNLayerList[LayerInd].FeatureMapList[d].OutputSS[b].shape[0],
                                                                                           self.CNNLayerList[LayerInd].FeatureMapList[d].OutputSS[b].shape[1]))
                        for s in range (0, self.CNNLayerList[LayerInd].FeatureMapList[d].OutputSS[b].shape[0]):
                            for z in range(0, self.CNNLayerList[LayerInd].FeatureMapList[d].OutputSS[b].shape[1]):
                                self.CNNLayerList[LayerInd].FeatureMapList[d].DeltaSS[b,s,z] = deltaSSFlat[index,0]
                                index= index+1
                        #print("Deltas ",self.CNNLayerList[LayerInd].FeatureMapList[d].DeltaSS)

                    #process CNN layers in reverse order, from last layer towards input
                    for cnnCount in range ( len(self.CNNLayerList)-1, -1, -1):
                        #print("cnnCount: ", cnnCount)
                        #compute deltas on the C layers - distrbute deltas from SS layer
                        #then multiply by the activation function
                        for k in range (0, len(self.CNNLayerList[cnnCount].FeatureMapList)):
                            fmp = self.CNNLayerList[cnnCount].FeatureMapList[k]
                            indexm = 0
                            indexn = 0
                            fmp.DeltaCV[b] = np.zeros((fmp.OutputSS[b].shape[0] * 2, fmp.OutputSS[b].shape[1] * 2))
                            for m in range( 0 , fmp.DeltaSS[b].shape[1]):
                                indexn = 0
                                for n in range( 0 , fmp.DeltaSS[b].shape[1]):
                                    if (fmp.activationType == ActivationType.SIGMOID):
                                        fmp.DeltaCV[b, indexm , indexn] = (1 / 4.0) * fmp.DeltaSS[b, m, n] * fmp.APrime[b, indexm, indexn]
                                        fmp.DeltaCV[b, indexm, indexn + 1] = (1 / 4.0) * fmp.DeltaSS[b, m, n] * fmp.APrime[b, indexm, indexn + 1]
                                        fmp.DeltaCV[b, indexm + 1, indexn] = (1 / 4.0) * fmp.DeltaSS[b, m, n] * fmp.APrime[b, indexm + 1, indexn]
                                        fmp.DeltaCV[b, indexm + 1, indexn + 1] = (1 / 4.0) * fmp.DeltaSS[b, m, n] * fmp.APrime[b, indexm + 1, indexn + 1]
                                        indexn = indexn + 2
                                    if (fmp.activationType == ActivationType.RELU):
                                        if (fmp.Sum[b, indexm, indexn] > 0):
                                            fmp.DeltaCV[b, indexm, indexn] = (1 / 4.0) * fmp.DeltaSS[b, m, n]
                                        else:
                                            fmp.DeltaCV[b, indexm, indexn] = 0

                                        if (fmp.Sum[b, indexm, indexn + 1] > 0):
                                            fmp.DeltaCV[b, indexm, indexn + 1] = (1 / 4.0) * fmp.DeltaSS[b, m, n]
                                        else:
                                            fmp.DeltaCV[b, indexm, indexn + 1] = 0

                                        if (fmp.DeltaCV[b, indexm + 1, indexn] > 0):
                                            fmp.DeltaCV[b, indexm + 1, indexn] = (1 / 4.0) * fmp.DeltaSS[b, m, n]
                                        else:
                                            fmp.DeltaCV[b, indexm + 1, indexn] = 0

                                        if (fmp.DeltaCV[b, indexm + 1, indexn + 1] > 0):
                                            fmp.DeltaCV[b, indexm + 1, indexn + 1] = (1 / 4.0) * fmp.DeltaSS[b, m, n]
                                        else:
                                            fmp.DeltaCV[b, indexm + 1, indexn + 1] = 0
                                        indexn = indexn + 2
                                indexm = indexm + 2
                        #print("DeltaCV[b]: ", fmp.DeltaCV[b])
                        #----------compute BiasGrad in current CNN Layer-------
                        for q in range (len(self.CNNLayerList[cnnCount].FeatureMapList)):
                            for u in range( 0, self.CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV[b].shape[1]):
                                for v in range ( 0,  self.CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV[b].shape[1]):
                                    self.CNNLayerList[cnnCount].FeatureMapList[q].BiasGrad += self.CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV[b, u, v]
                        #print("fmp..BiasGrad: ", fmp.BiasGrad)
                        #----------compute gradients for pxq kernels in current CNN layer--------
                        if (cnnCount > 0):  # not the first CNN layer
                            for p in range ( 0, len(self.CNNLayerList[cnnCount - 1].FeatureMapList)):
                                for q in range ( 0, len(self.CNNLayerList[cnnCount].FeatureMapList)):
                                    RotatedOutPutSS = self.RotateBy90(self.RotateBy90(self.CNNLayerList[cnnCount - 1].FeatureMapList[p].OutputSS[b]))
                                    self.CNNLayerList[cnnCount].KernelGrads[p, q] = self.CNNLayerList[cnnCount].KernelGrads[p, q] + signal.convolve2d(RotatedOutPutSS, self.CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV[b], mode = 'valid')
                                #print("self.CNNLayerList[cnnCount - 1].FeatureMapList[p].OutputSS[b]): ", self.CNNLayerList[cnnCount - 1].FeatureMapList[p].OutputSS[b])
                                #print("RotatedOutPutSS", RotatedOutPutSS)
                            #---------------this layer is done, now backpropagate to prev CNN Layer----------
                            for p in range ( 0 , len(self.CNNLayerList[cnnCount - 1].FeatureMapList)):
                                size = self.CNNLayerList[cnnCount - 1].FeatureMapList[p].OutputSS[b].shape[0]
                                self.CNNLayerList[cnnCount - 1].FeatureMapList[p].DeltaSS[b] = np.zeros((size, size))
                                for q in range (0, len(self.CNNLayerList[cnnCount].FeatureMapList)):
                                    RotatedKernel = self.RotateBy90(self.RotateBy90(self.CNNLayerList[cnnCount].Kernels[p, q]))
                                    self.CNNLayerList[cnnCount - 1].FeatureMapList[p].DeltaSS[b] = self.CNNLayerList[cnnCount - 1].FeatureMapList[p].DeltaSS[b] + signal.convolve2d(self.CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV[b], RotatedKernel, mode='full')
                                
                        else:  #very first CNN layer which is connected to input
                            #has 1xnumFeaturemaps 2-D array of Kernels and Kernel Gradients
                            #----------compute gradient for first layer cnn kernels--------
                            for p in range (0 , 1):
                                for q in range( 0, len(self.CNNLayerList[cnnCount].FeatureMapList)):
                                    RotatedInput = self.RotateBy90(self.RotateBy90(self.InputDataList[dj + b]))
                                    self.CNNLayerList[cnnCount].KernelGrads[p, q] = self.CNNLayerList[cnnCount].KernelGrads[p, q] + signal.convolve2d(RotatedInput, self.CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV[b], mode='valid')
                        #print("self.CNNLayerList[cnnCount].KernelGrads: ", self.CNNLayerList[cnnCount].KernelGrads)

                trainingError += sum(trainingErr)
                self.UpdateKernelsWeightsBiases(learningRate, batchSize)
                self.ClearGradients()

            if (i % 10 == 0):
                learningRate = learningRate / 2  #reduce learning rate
            print("epoch = " , i , " training error = " , trainingError)

    def UpdateKernelsWeightsBiases(self, learningRate, batchSize):
        #print("Helloww from update")
        #---------------update kernels and weights-----
        for cnnCount in range (0 , len(self.CNNLayerList)):
            if (cnnCount == 0):  # first CNN layer
                for p in range (0 , 1):
                    for q in range (0 , len(self.CNNLayerList[0].FeatureMapList)):
                        self.CNNLayerList[cnnCount].Kernels[p, q] = self.CNNLayerList[cnnCount].Kernels[p, q] - self.CNNLayerList[cnnCount].KernelGrads[p, q] * (1.0 / batchSize) * (learningRate)

            else: # next CNN layers
                for p in range (0, len(self.CNNLayerList[cnnCount-1].FeatureMapList)):
                    for q in range ( 0, len(self.CNNLayerList[cnnCount].FeatureMapList)):
                        self.CNNLayerList[cnnCount].Kernels[p, q] = self.CNNLayerList[cnnCount].Kernels[p, q] - self.CNNLayerList[cnnCount].KernelGrads[p, q] * (1.0 / batchSize) * (learningRate)

            for d in range (len(self.CNNLayerList[cnnCount].FeatureMapList)):
                self.CNNLayerList[cnnCount].FeatureMapList[d].Bias = self.CNNLayerList[cnnCount].FeatureMapList[d].Bias - (self.CNNLayerList[cnnCount].FeatureMapList[d].BiasGrad / batchSize) * learningRate
        for d in range (len(self.LayerList)):
            GradW = np.zeros((self.LayerList[d].GradW.shape[1], self.LayerList[d].GradW.shape[2]))
            for b in range ( 0, batchSize):
                GradW = GradW + self.LayerList[d].GradW[b]

            GradB = np.zeros((self.LayerList[d].numNeurons, 1))
            for b in range( 0, batchSize):
                GradB = GradB + self.LayerList[d].GradB[b]
            self.LayerList[d].W = self.LayerList[d].W - GradW * (1.0 / batchSize*learningRate)
            self.LayerList[d].B = self.LayerList[d].B - GradB * (1.0 / batchSize*learningRate)

    def ClearGradients(self):
        #print("hello from clear")
        for cnnCount in range( 0 , len(self.CNNLayerList)):
            if (cnnCount == 0):  # first CNN layer
                for p in range (0 , 1):
                    for q in range(0, len(self.CNNLayerList[cnnCount].FeatureMapList)):
                        self.CNNLayerList[cnnCount].KernelGrads[p, q]= np.zeros((self.CNNLayerList[cnnCount].kernelSize,self.CNNLayerList[cnnCount].kernelSize))
            else: # next CNN layers
                for p in range ( 0, len(self.CNNLayerList[cnnCount - 1].FeatureMapList)):
                    for q in range ( 0 , len(self.CNNLayerList[cnnCount].FeatureMapList)):
                        self.CNNLayerList[cnnCount].KernelGrads[p, q]=np.zeros((self.CNNLayerList[cnnCount].kernelSize,self.CNNLayerList[cnnCount].kernelSize))
            for d in range (len(self.CNNLayerList[cnnCount].FeatureMapList)):
                self.CNNLayerList[cnnCount].FeatureMapList[d].BiasGrad = 0

            for d in range (len(self.LayerList)):
                for b in range (0 , self.LayerList[d].batchSize):
                    self.LayerList[d].GradW[b]= np.zeros((self.LayerList[d].numNeurons,self.LayerList[d].inputSize))
                    self.LayerList[d].GradB[b]= np.zeros((self.LayerList[d].numNeurons,1))

    def RotateBy90(self, a):
        Res = np.zeros((a.shape[0], a.shape[1]))
        for i in range ( 0, a.shape[0]):
            for j in range ( 0, a.shape[1]):
                Res[i, j] = a[a.shape[0] - j - 1][i]
        return Res
