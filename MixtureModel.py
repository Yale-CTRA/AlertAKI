#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:41:49 2018

@author: aditya
"""

import numpy as np
import torch
from torch.nn import init
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn import Parameter


class Classifier1(nn.Module):
    def __init__(self, inputSize, mixtureSize):
        super().__init__()
        self.linearPre = nn.Linear(inputSize, 12)
        init.kaiming_normal(self.linearPre.weight)
        init.constant(self.linearPre.bias, 0)
        
        self.linear = nn.Linear(12, mixtureSize)
        init.kaiming_normal(self.linear.weight)
        init.constant(self.linear.bias, 0)
        
        
        self.linearMixPre = nn.Linear(inputSize, 12)
        init.kaiming_normal(self.linearMixPre.weight)
        init.constant(self.linearMixPre.bias, 0)
        
        self.linearMix = nn.Linear(12, mixtureSize)
        init.kaiming_normal(self.linearMix.weight)
        init.constant(self.linearMix.bias, 0)
        
        
    def forward(self, input):
        if self.training:
            input = F.dropout(input, 0.2)
        y_mix = F.softmax(self.linearMix(F.tanh(self.linearMixPre(input))), dim = 1)
        y_all = self.linear(F.tanh(self.linearPre(input)))
        y = torch.sum(y_mix*y_all, dim = 1)
        return y, y_mix


class Classifier2(nn.Module):
    def __init__(self, inputSize, mixtureSize):
        super().__init__()
        #self.bn = nn.BatchNorm1d(inputSize)
        self.linear = nn.Linear(inputSize, 1)
        init.kaiming_normal(self.linear.weight)
        init.constant(self.linear.bias, 0)
    
        
    def forward(self, input):
        y = self.linear(input)
        return y, None


class MixtureUplift(nn.Module):
    def __init__(self, inputSize):
        super().__init__()
        self.linear = nn.Linear(inputSize, 10)
        self.linear2 = nn.Linear(10, 5)
        self.linear3 = nn.Linear(3, 1)
        
        init.xavier_uniform(self.linear.weight)
        init.kaiming_uniform(self.linear2.weight)
        
    def noise(self, input, p = 0.33):
        std = p/(1-p)
        if self.training:
            input += Variable(input.data, requires_grad = False)*Variable(torch.cuda.FloatTensor(input.size()).normal_(mean = 0, std = std))
        return input
        
    def forward(self, X, A):
        output = self.linear2(F.tanh(self.noise(self.linear(X))))
        output = self.noise(output)
        y = self.linear3(F.selu(output[:,2:]))[:,0]
        mix = F.sigmoid(output[:,0])
        deltaPre = mix*output[:,1]
        delta = deltaPre*A
        
        y = F.relu(y) - delta
        return y, deltaPre


class MixtureUpliftSimple(nn.Module):
    def __init__(self, inputSize, ate):
        super().__init__()
        
        self.ate = ate
        self.linear = nn.Linear(inputSize, 24)
        init.xavier_normal(self.linear.weight)
        self.linear2 = nn.Linear(24, 2)
        self.linear3 = nn.Linear(inputSize, 1)
        
    def noise(self, input, p = 0.1):
        std = p/(1-p)
        if self.training:
            input += Variable(input.data.clone(), requires_grad = False)*Variable(torch.cuda.FloatTensor(input.size()).normal_(mean = 0, std = std))
        return input
        
    def forward(self, input, A):
        m = A.size()[0]
        
        output = self.noise(self.linear2(F.tanh(self.noise(self.linear(input), p = 0.25))), p = 0.2)
        mix = F.sigmoid(output[:,0])
        deltaPre = mix*self.noise(self.linear3(input)[:,0], p = 0.2)
        delta = (deltaPre - self.ate)*A
        
        y = F.relu(output[:,1] - delta)
        return y, deltaPre






class MixLearner(object):
    def __init__(self, alpha = 1e-3, ate = 0):
        super().__init__()
        self.fitted = False
        self.alpha = alpha
        self.epochsPassed = 0
        self.ate = ate
        
    def fit(self, X, Y, A, epochs = 100, batchSize = 16, lr = 1e-2, verbose = False):
        if not self.fitted:
            inputSize = np.shape(X)[1]
            self.model = MixtureUpliftSimple(inputSize, self.ate).cuda()
            
        self.epochsPassed += epochs
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr = lr, weight_decay = self.alpha)
        dataLoader = DataLoader(dataset = TensorDataset(torch.from_numpy(X).cuda(), 
                                                        torch.from_numpy(np.concatenate((Y[:,None], A[:,None]),
                                                            axis = 1)).cuda()), batch_size = batchSize, shuffle = True)

        for epoch in range(epochs):
            losses, batchNum = 0, 0
            for batch in dataLoader:
                X, Y = batch
                X, Y, A= Variable(X), Variable(Y[:,0]), Variable(Y[:,1])
                y_est, deltaPre = self.model(X, A)
                loss = F.mse_loss(y_est, Y)  + torch.mean(3e-2*F.relu(deltaPre) + 3e-2*F.relu(-deltaPre))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.data[0]
                batchNum += 1
            
            if verbose:
                if epoch % 2 == 0:
                    print(losses/batchNum)
                
        self.fitted = True
    
    def predict(self, X, switch = False):
        assert self.fitted
        X = Variable(torch.from_numpy(X)).cuda()
        A = Variable(torch.cuda.FloatTensor(len(X)))
        self.model.train()
        U = np.zeros((len(X), 1000), dtype = np.float32)
        for i in range(1000):
            U[:,i] = self.model(X, A)[1].cpu().data.numpy()
            
        U = np.mean(U, axis = 1)
        return -U if switch else U


ATE = lambda Y, A: np.mean(Y[A==1]).item() - np.mean(Y[A==0]).item()

predictors = predictorBase + predictorsImmune

allPredictions = pd.DataFrame(index = df.index)
allPredictions[target] = df[target]
allPredictions[action] = df[action]


iterations = 10
trainIndex, valIndex, testIndex = makeSetIndices(df, 0.5, 0.2)
trainValIndex = trainIndex + valIndex
auucs = []
counter = 0
while counter < iterations:
    X, Y, A = refresh(temporal = False)
    xTrainVal, xTest = X
    xTrainVal, xTest = xTrainVal[predictors], xTest[predictors]
    yTrainVal, yTest = Y
    aTrainVal, aTest = A 
    
    ## training loop
    model = MixLearner(alpha = 1e-3, ate = ATE(yTrainVal, aTrainVal))
    condition = True
    best, noImprovement = 1e8, 0
    epochsPassed = 0
    while condition:
        if epochsPassed < 150:
            model.fit(xTrainVal.values.astype(np.float32), yTrainVal, aTrainVal, epochs = 25, lr = 7e-2, verbose = True)
        else:
            model.fit(xTrainVal.values.astype(np.float32), yTrainVal, aTrainVal, epochs = 25, lr = 3e-2, verbose = True)
        epochsPassed += 25
        if epochsPassed > 200:
            uTrainVal = model.predict(xTrainVal.values.astype(np.float32))
            performance = AUUC(uTrainVal, yTrainVal, aTrainVal, graph = False)
            if performance < best:
                torch.save(model.model, 'model.pt')
                best = performance
                noImprovement = 0
            else:
                noImprovement += 1
            if noImprovement == 4:
                condition = False
            
            
    ## evaluation
    model.model = torch.load('model.pt')
    uTest = model.predict(xTest.values.astype(np.float32))
    auuc = AUUC(uTest, yTest, aTest, graph = True)
    print(auuc)
    #auuc = AUUC(uTest, yTest, aTest, graph = True)
    
    ##record
    auucs += [auuc]
    uTest = pd.DataFrame(uTest[:,None], index = xTest.index)
    allPredictions[counter] = uTest
    print(counter)
    counter += 1
