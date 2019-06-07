# File Name: Assignment5.py
# Author: Zainab Alaweel
# Date: 04 /20 / 2019
import sys
import os
import cv2
from ActivationType import ActivationType
from PoolingType import PoolingType
from FeatureMap import *
from CNNLayer import *
from Layer import *
from DeepCNN import *
import numpy as np

def main():
    TestList =[]
    LablesList =[]
    batchSize = 1
    numFeatureMapsLayer1 = 6 
    CNNList = []
    NNLayerList = []
    numFeatureMapsLayer2 = 12 
    C1 = CNNLayer(numFeatureMapsLayer1, 1, 28, 5,  ActivationType.RELU, PoolingType.AVGPooling, batchSize) 
    C2 = CNNLayer(numFeatureMapsLayer2, numFeatureMapsLayer1, 12, 5, ActivationType.RELU, PoolingType.AVGPooling, batchSize)
    CNNList.append(C1)
    CNNList.append(C2)
    l1 = Layer(50, 4 * 4 * numFeatureMapsLayer2, ActivationType.RELU,batchSize,0.8)
    l2 = Layer(10, 50, ActivationType.SOFTMAX,batchSize)
    NNLayerList.append(l1)
    NNLayerList.append(l2)

    InputTrainingList, OutputTrainingLabels  = ReadMNISTTrainingData()
    InputTestingList, OutputTestingLabels = ReadMNISTTestingData()
    dcnn = DeepCNN(CNNList, NNLayerList, InputTrainingList, OutputTrainingLabels ,batchSize)
    dcnn.Train(30, 0.1, batchSize) 

    accuracyCount = 0
    for i in range(len(InputTestingList)):
        # do forward pass
        res = dcnn.Evaluate(InputTestingList[i],0)
        # determine index of maximum output value
        maxindex = res.argmax(axis = 0)
        if (OutputTestingLabels[i][maxindex] == 1):
            accuracyCount = accuracyCount + 1
    print("done training.., accuracy = " , ( accuracyCount /len(InputTestingList) * 100))

def ReadMNISTTrainingData(): # reading training data
    InputList = []
    OutputLabels = []
    train = np.empty((28,28),dtype='float64')
    trainY = np.zeros((10,1))
    for filename in os.listdir('C:/Users/Zainab Altaweel/source/repos/Assignment5/Data/Training1000/'):
        y = int(filename[0])
        trainY[y] = 1.0
        train = cv2.imread('C:/Users/Zainab Altaweel/source/repos/Assignment5/Data/Training1000/{0}'.format(filename),0)/255.0 
        InputList.append(train)
        OutputLabels.append(trainY)
        trainY = np.zeros((10,1))
    return InputList, OutputLabels

def ReadMNISTTestingData(): # reading testing data
    InputList = []
    OutputLabels = []
    train = np.empty((28,28),dtype='float64')
    trainY = np.zeros((10,1))
    for filename in os.listdir('C:/Users/Zainab Altaweel/source/repos/Assignment5/Data/Test10000/'):
        y = int(filename[0])
        trainY[y] = 1.0
        train = cv2.imread('C:/Users/Zainab Altaweel/source/repos/Assignment5/Data/Test10000/{0}'.format(filename),0)/255.0 
        InputList.append(train)
        OutputLabels.append(trainY)
        trainY = np.zeros((10,1))
    return InputList, OutputLabels

if __name__ == "__main__":
    sys.exit(int(main() or 0))
"""TestList =[]
LablesList =[]
for i in range (4):
    M = np.random.uniform(low = 0, high=1, size=(28,28))
    TestList.append(M)
    N = np.ones((10,1))
    LablesList. append(N)
#print("TestList ", TestList)
#TestFeatureMap = FeatureMap(12, ActivationType.RELU , PoolingType.MAXPooling ,1)
#TestFeatureMap.Evaluate(M,1)

TestCNNList = []
TestCNNLayer1 = CNNLayer(4,1,28,5,PoolingType.AVGPooling, ActivationType.SIGMOID,1)
TestCNNLayer2= CNNLayer(6,4,12,5,PoolingType.AVGPooling, ActivationType.SIGMOID,1)

TestCNNList.append(TestCNNLayer1)
TestCNNList.append(TestCNNLayer2)
#TestCNNLayer.Evaluate(TestList,0)

TestLayerList = []
TestLayer1 = Layer(50,96,ActivationType.SIGMOID,1,1.0,0.8)
TestLayer2 = Layer(10,50,ActivationType.SIGMOID,1,1.0,0.8)
TestLayerList.append(TestLayer1)
TestLayerList.append(TestLayer2)
#TestLayer.Evaluate(M,0,True,True)


TestDeepCNN = DeepCNN(TestCNNList, TestLayerList, TestList, LablesList, 1)
TestDeepCNN.Train(5, 0.01, 1)
#TestDeepCNN.Evaluate(M, 0)

#Test rotate 
#A=np.array([[1,2,3],[4,5,6],[7,8,9]])
#print("A: ", A)
#print("Rotated A: ", TestDeepCNN.RotateBy90(A))"""