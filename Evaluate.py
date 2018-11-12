#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 21:22:56 2017

@author: aditya
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler, MinMaxScaler
import os
from joblib import Parallel, delayed

#os.chdir('/home/aditya/Projects/AKI Alert/Code/first/')
from helper import AUUC
os.chdir('/home/aditya/Projects/AKI Alert/Code/second/')
from Models import TLearner, XLearner, ZLearner, ProgLearner
#os.chdir('/home/aditya/Projects/AKI Alert/Data/')




action = 'assignment'
#target = 'yLastPer'
target = 'yMaxPer'

predictorBase = 'cratio0 cratio1 mcv mchc icuatalert c0value c1value bicarbonate bun age malegender hemoglobin wbc plateletcount sodium surgical'.split(' ')
predictorBank = ['uaprotein', 'timesec', 'redcelldistribution', 'alkphos', 'cdeltapercent', 'cslope',
                  'aaorno', 'pt', 'eospercent',  'neutrophilabs', 'basosabs','uaspecgrav',
                  'bilitotal', 'magnesium', 'missing', 'orders','chloride', 'glucose',
                  'hematocrit', 'mch', 'lactate', 'phosphorus','chf', 'basophilpercent',
                  'lymphpercent', 'leukocyteabs', 'neutrophilpercent', 'monopercent', 'monoabs', 'eosinophilabs', 'potassium',
                  'nsaidcategory', 'antibioticcategory', 'narcoticcategory', 'pressorcategory']
predictorBank += ['paralyticcategory']
#predictorBank += ['loopcategory', 'acearbcategory','anycontrastpre']
predictorsAll = predictorBase + predictorBank

predictorsBinary = ['antibioticcategory', 'narcoticcategory', 
                    'pressorcategory', 'nsaidcategory', 'aaorno', 'icuatalert',
                    'malegender', 'surgical', 'chf']
predictorsBinary += ['paralyticcategory']


percentPredictors = ['eospercent',  'neutrophilpercent', 'monopercent', 'lymphpercent', 'basophilpercent']
#predictorsBinary += ['anycontrastpre', 'loopcategory', 'acearbcategory',]


predictorsFloat = list(set(predictorsAll) - set(predictorsBinary + ['timesec']))
predictorsImmune = ['eospercent',  'neutrophilpercent', 'monopercent', 'lymphpercent', 'basophilpercent', 
                    'monoabs', 'eosinophilabs', 'leukocyteabs',  'basosabs', 'neutrophilabs', 'redcelldistribution']

def makeSetIndices(df, trainPercent, valPercent):
    trainIndex = np.repeat(False, len(df))
    valIndex = np.repeat(False, len(df))
    testIndex = np.repeat(False, len(df))
    
    cutTrain = int(round(len(df)*trainPercent))
    cutVal = int(round(len(df)*(trainPercent + valPercent)))
    
    trainIndex[:cutTrain] = True
    valIndex[cutTrain:cutVal] = True
    testIndex[cutVal:] = True
    
    return trainIndex, valIndex, testIndex

def refresh(temporal = False):
    global df, trainValIndex, testIndex, predictorBase, predictorBank
    if temporal is True:
        df = df.sort_values(by = 'timesec')
    else:
        df = df.sample(frac = 1.)
        
    dfTrain = df.loc[trainValIndex,:]
    dfTest = df.loc[testIndex,:]
    imputerFloat = Imputer()
    imputerBinary = Imputer(strategy = 'most_frequent')
    scaler = RobustScaler()
    
    xTrainFloat = imputerFloat.fit_transform(dfTrain[predictorsFloat].values)
    xTestFloat = imputerFloat.transform(dfTest[predictorsFloat].values)
    xTrainBinary = imputerBinary.fit_transform(dfTrain[predictorsBinary].values)
    xTestBinary = imputerBinary.transform(dfTest[predictorsBinary].values)
    
    xTrainFloat = scaler.fit_transform(xTrainFloat)
    xTestFloat = scaler.transform(xTestFloat)

    
    xTrain = np.concatenate((dfTrain['timesec'].values[:,None], xTrainBinary, xTrainFloat), axis = 1)
    xTest= np.concatenate((dfTest['timesec'].values[:,None], xTestBinary, xTestFloat), axis = 1)
    xTrain = pd.DataFrame(xTrain, columns = ['timesec'] + predictorsBinary  + predictorsFloat, index = dfTrain.index)
    xTest = pd.DataFrame(xTest, columns = ['timesec'] + predictorsBinary  + predictorsFloat, index = dfTest.index)
    
    yTrain, yTest = dfTrain[target].values.astype(np.float32), dfTest[target].values.astype(np.float32)
    aTrain, aTest = dfTrain[action].values.astype(np.float32), dfTest[action].values.astype(np.float32)
    
    return (xTrain, xTest), (yTrain, yTest), (aTrain, aTest)


def featureSelect(modelClass, xTrain, xVal, yTrain, yVal, aTrain, aVal):
    global predictorBase, predictorBank, alpha
    predictorsCurrent = predictorBase.copy()
    searching = True
    bestPerformance = 1e8
    while searching:
        bestSubPerformance = 1e8
        for k in range(len(predictorBank)):
            if predictorBank[k] not in predictorsCurrent:
                predictors = predictorsCurrent + [predictorBank[k]]
                model = modelClass(alpha = alpha)
                model.fit(xTrain[predictors].values.astype('float32'), yTrain, aTrain)
                
                uVal = model.predict(xVal[predictors].values.astype('float32'))
                
                if modelClass is ProgLearner:
                    auuc = np.mean(np.power(uVal - yVal, 2))
                else:
                    auuc = AUUC(uVal, yVal, aVal, graph = False)
        
                if auuc < bestSubPerformance:
                    currentFeature = predictorBank[k]
                    bestSubPerformance = auuc
                
        if bestSubPerformance < bestPerformance:
            bestPerformance = bestSubPerformance
            predictorsCurrent += [currentFeature]
        else:
            searching = False
    
    return predictorsCurrent


def train(xTrainVal, yTrainVal, aTrainVal, xTest, index):
    global trainIndex, valIndex, shuffleIndicesAll, alpha
    shuffleIndices = shuffleIndicesAll[index]
    
    xTrainVal, yTrainVal, aTrainVal = xTrainVal.iloc[shuffleIndices,:], yTrainVal[shuffleIndices], aTrainVal[shuffleIndices]
    xTrain, xVal = xTrainVal.iloc[trainIndex,:], xTrainVal.iloc[valIndex,:]
    yTrain, yVal = yTrainVal[trainIndex], yTrainVal[valIndex]
    aTrain, aVal = aTrainVal[trainIndex], aTrainVal[valIndex]
    
    ## full forward feature selection process
    predictorList = featureSelect(modelClass, xTrain, xVal, yTrain, yVal, aTrain, aVal)
    
    
    model = modelClass(alpha = alpha)
    ## train and predict with final features
    model.fit(xTrain[predictorList].values.astype('float32'), yTrain, aTrain)
    return(model.predict(xTest[predictorList].values.astype('float32'))[:,None], predictorList, model)


baseStr = 'LearnerMaxTemporalRand.csv'
saveNameDict = {XLearner: 'X'+baseStr, ZLearner: 'Z'+baseStr, TLearner: 'T'+baseStr, 
            ProgLearner: 'Prog'+baseStr}


randomTreatments = True
temporal = True
predictEverything = False
iterations = 100
cvIters = 21
alpha = 0

for modelClass in [ZLearner, XLearner, TLearner, ProgLearner]:
    allPredictions = pd.DataFrame(index = df.index)
    allPredictions[target] = df[target]
    allPredictions[action] = df[action]
    allRandomTreatments = pd.DataFrame(index = df.index)
    
    trainIndex, valIndex, testIndex = makeSetIndices(df, 0.5, 0.2)
    trainValIndex = trainIndex + valIndex
    trainIndex, valIndex = trainIndex[:sum(trainValIndex)], valIndex[:sum(trainValIndex)]
    #modelClass = ZLearner
    
    auucs = []
    counter = 0
    while counter < iterations:
        X, Y, A = refresh(temporal = temporal)
        xTrainVal, xTest = X
        yTrainVal, yTest = Y
        if randomTreatments:
            aTrainVal = np.random.choice([True, False], len(yTrainVal))
            aTest = np.random.choice([True, False], len(yTest))
            allRandomTreatments[counter] = pd.Series(np.concatenate((aTrainVal,aTest)),
                                                       index = np.concatenate((xTrainVal.index.values, xTest.index.values)))
        else:
            aTrainVal, aTest = A 

        
        ## cross validation loop parallelized
        shuffleIndicesAll = [np.random.permutation(range(len(yTrainVal))) for i in range(cvIters)]
        results = Parallel(n_jobs = 7)(delayed(train)(xTrainVal, yTrainVal, aTrainVal == 1, xTest, i) for i in range(cvIters))
        uTest, predictorList, modelList = list(zip(*results))
        uTest = np.concatenate(uTest, axis = 1)
        
        ## turn to true for predicting everything for this cut
        if predictEverything:
            xAll = pd.concat([xTrainVal, xTest], axis = 0)
            A = np.concatenate((aTrainVal, aTest), axis = 0)
            U = np.concatenate([modelList[j].predict(xAll[predictorList[j]].values)[:,None] for j in range(cvIters)], axis = 1)
            Y = np.concatenate((yTrainVal, yTest), axis = 0)
            
            results = pd.DataFrame(data = {'uplift': np.mean(U, axis = 1), 'testing': testIndex, 
                                        'alert': A, 'yMaxPer': Y}, index = xAll.index)
            results.to_csv('../../Data/results/' + saveNameDict[modelClass])
        
        
        
        ## evaluate and add performance to list
        auuc = AUUC(np.mean(uTest, axis = 1), yTest, aTest, graph = True)
        # below only works after running predictEverything block
        #auuc = AUUC(np.mean(U[trainValIndex,:], axis = 1), Y[trainValIndex], A[trainValIndex], graph = False)
        auucs += [auuc]
        
        
        # below line for a fully randomly generated set of predictions
        # useful for exploring variance
        #uTest = pd.DataFrame(np.random.uniform(-10, 10, size = len(yTest)), index = xTest.index)
        
        ## record test set predictions
        uTest = pd.DataFrame(np.mean(uTest, axis = 1), index = xTest.index)
        allPredictions[counter] = uTest
        
        print(counter)
        counter += 1
        
        
            
    
    print('\nMean: ' + str(np.mean(auucs)))
    print('Std: '+ str(np.std(auucs)))

        
    allPredictions.to_csv('../../Data/results/' + saveNameDict[modelClass])
    allRandomTreatments.to_csv('../../Data/results/RandAlerts_' + saveNameDict[modelClass])
        
        
        

        