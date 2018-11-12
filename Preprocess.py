#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:54:43 2017

@author: aditya
"""


from scipy.stats import skewtest, boxcox
import numpy as np
import seaborn as sns
import pandas as pd
import os
    
os.chdir('/home/aditya/Projects/AKI Alert/Code/second/')
#os.chdir('C:\\Users\\adity\\Projects\\AKI-Alert')

from helper import loadDict, searchName, createDict
    
def isSkewed(df, vars, significance = 1e-5):
    results = [False for i in range(len(vars))]
    for i, var in enumerate(vars):
        data = df.loc[np.isfinite(df[var]),var].values
        if len(np.unique(data)) > 2:
            results[i] = skewtest(data)[1] < significance
    return results


def logTrans(df, names):
    for name in names:
        restrictedColumn = df.loc[np.isfinite(df[name]), name].values
        df.loc[np.isfinite(df[name]),name] =  np.log(restrictedColumn + 1)
    return df

def boxcoxTrans(df, names):
    for name in names:
        restrictedColumn = df.loc[np.isfinite(df[name]), name].values
        df.loc[np.isfinite(df[name]),name] =  boxcox(restrictedColumn - np.min(restrictedColumn) + 1)[0]
    return df

os.chdir('/home/aditya/Projects/AKI Alert/Data/')
#dataFileName = '3 day outcome all patients.dta'
dataFileName = 'full dataset two outcomes no predictions.dta'


df = pd.read_stata(dataFileName)
df.index = df['uid_id'].values.astype('int64')
df['timesec'] = pd.to_numeric(df['timesec'], errors = 'coerce')
df['assignment'] = 1*(df['assignment'] == 'Alert')
df = df[df['akinstage'] == 1]



## create new input vars
df['creatchange'] = df['c1value'] - df['c0value']
df['cdeltapercent'] = df['creatchange']/df['c0value']
df['cslope'] = df['creatchange']/df['creat0tocreat1time']
df['cratio'] = df['c1value']/df['c0value']

## create outcome vars
baseline = df['c1value']
df['yLast'] = df['lastcreat72']
df['yLastPer'] = (df['yLast'] - baseline)/baseline
df['yMax'] = df['maxcreat72']
df['yMaxPer'] = (df['yMax'] - baseline)/baseline
#df['Z'] = pd.Series(np.zeros(len(df)), index = df.index)
#df.loc[df['assignment'] == 1, 'Z'] = df.loc[df['assignment'] == 1, 'creatoutcome7percent']
#df.loc[df['assignment'] == 0, 'Z'] = -df.loc[df['assignment'] == 0, 'creatoutcome7percent']
#df.loc[df['creatoutcome7percent'] == 0, 'creatoutcome7percent'] = (df.loc[df['creatoutcome7percent'] == 0, 'c0value'] -   \
#       df.loc[df['creatoutcome7percent'] == 0, 'c1value'])/df.loc[df['creatoutcome7percent'] == 0, 'c1value']  


df['cratio0'] = df['icuatalert']*df['cratio']
df['cratio1'] = (1-df['icuatalert'])*df['cratio']


## load dictionary, flatten, and remove medications, missing vars, etc...
#varDict = loadDict('multiview alert.csv', df)


# dont know why i have to do this, but for some reason the uplift package cant find the feature 'assignment' w/o it....
#df['missing'] = 6
df['missing'] = np.sum(np.isfinite(df.loc[:,'a1c':'wbc'].values), axis = 1)


df['orders'] = df.loc[:,'ablationorder':'ventpcorder'].sum(axis = 1)
#varDict['orders'] = ['missing', 'orders', 'timesec']
#predictors = [item for sublist in varDict.values() for item in sublist]
predictors = ['timesec', 'creatchange', 'cdeltapercent', 'cslope', 'cratio', 'c1value', 'c0value', 'age', 'aaorno', 'surgical',
          'malegender', 'mcv', 'bun', 'bilidirect', 'redcelldistribution', 'alkphos', 'hemoglobin', 'orders',
          'wbc', 'nsaidcategory', 'potassium', 'chloride', 'glucose', 'mchc', 'plateletcount', 'sodium', 'missing', 'icuatalert',
         'pressorcategory', 'uaspecgrav', 'uaprotein', 'alt', 'ast', 'pt', 'ptt', 'inr', 'hematocrit', 'mch',
         'chf', 'antibioticcategory', 'narcoticcategory',
         'basophilpercent', 'basosabs',
         'lymphpercent', 'leukocyteabs', 'neutrophilabs', 'neutrophilpercent', 'monopercent', 'monoabs', 'eosinophilabs',
         'eospercent', 'cratio0', 'cratio1',
         'bicarbonate', 'bilitotal', 'magnesium', 'lactate', 'phosphorus']
predictors += ['paralyticcategory']



## scale time
minTime, maxTime = np.min(df['timesec'].values), np.max(df['timesec'].values)
df['timesec'] = (df['timesec'] - minTime)/(maxTime - minTime)


## fix nans in neutrophilpercent
examine = np.logical_and(np.isnan(df['neutrophilpercent'].values), np.isfinite(df['monopercent'].values))
colsAdd = ['eospercent', 'monopercent', 'lymphpercent', 'basophilpercent']
df.loc[examine, 'neutrophilpercent'] = 100 - np.sum(df.loc[examine,colsAdd].values, axis = 1)



percentPredictors = ['eospercent',  'neutrophilpercent', 'monopercent', 'lymphpercent', 'basophilpercent']

## log transform heavily skewed variables
def logitTransform(df, vars):
    for var in vars:
        recorded = np.isfinite(df[var].values)
        p = df.loc[recorded,var].values/100
        nearOne = 1*(p > 0.9)
        nearZero = 1*(p < 0.1)
        p -= 0.01*nearOne
        p += 0.01*nearZero
        df.loc[recorded,var] = np.log(p/(1-p))
    return df

old = False

if old:
    varsLogTrans = np.array(predictors)[isSkewed(df, predictors)]
    df = logTrans(df, varsLogTrans)
else:
    varsLogTrans = list(set(np.array(predictors)[isSkewed(df, predictors)]) - set(percentPredictors + ['timesec']))
    df = boxcoxTrans(df, varsLogTrans)
    df = logitTransform(df, percentPredictors)
    


targets = ['yLast', 'yLastPer', 'yMax', 'yMaxPer']
df = df[predictors + targets + ['assignment']]
os.chdir('/home/aditya/Projects/AKI Alert/Code/second/')


