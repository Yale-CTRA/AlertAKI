# Start: March 31, 2017
# Last: March 31, 2017

import os
import pandas as pd
import numpy as np
from scipy.stats import skewtest, boxcox
from sklearn.preprocessing import Imputer, RobustScaler, MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import random
import pickle
import warnings
import matplotlib.pyplot as plt
from collections import OrderedDict

#import torch
#from torch.autograd import Variable


import matplotlib.pyplot as plt
def AUUC(U, Y, A, bins = 50, graph = True):
    m = len(U)
    # small random noise to break ties
    U2 = U + np.random.normal(0, scale = 1e-6, size = len(U))
    
    d = np.stack((U2,Y,A), axis = 1)
    sortIndex = np.argsort(U2)[::-1]
    d = d[sortIndex]
    
    treated, control = d[d[:,2] == 1], d[d[:,2] == 0]
    treated = np.concatenate((treated, np.cumsum(treated[:,1])[:,None], np.arange(1, len(treated) + 1)[:,None]), axis = 1)
    control = np.concatenate((control, np.cumsum(control[:,1])[:,None], np.arange(1, len(control) + 1)[:,None]), axis = 1)
    d = np.concatenate((treated, control), axis = 0)
    sortIndex = np.argsort(d[:,0])[::-1]
    d = d[sortIndex]
    
    
    
    # perform search for indices to measure with
    cutIndices = np.round(np.linspace(0, m-1, bins + 1)).astype('int')[1:]
    treatedSums = np.zeros(len(cutIndices))
    controlSums = np.zeros(len(cutIndices))
    treatedM = np.zeros(len(cutIndices))
    controlM = np.zeros(len(cutIndices))
    

    for i,index in enumerate(list(cutIndices)):
        onTreated = True if d[index,2] == 1 else False
        if onTreated:
            treatedSums[i] = d[index,3]
            treatedM[i] = d[index,4]
            found = False
            counter = 0
            while not found:
                counter += 1
                if d[index - counter,2] == 0:
                    found = True
            controlSums[i] = d[index - counter,3]
            controlM[i] = d[index - counter,4]
        else:
            controlSums[i] = d[index,3]
            controlM[i] = d[index,4]
            found = False
            counter = 0
            while not found:
                counter += 1
                if d[index - counter,2] == 1:
                    found = True
            treatedSums[i] = d[index - counter,3]
            treatedM[i] = d[index - counter,4]
        
    treatedPercent = treatedSums/treatedM
    controlPercent = controlSums/controlM
    
    uplift = np.concatenate((np.zeros(1), 0.5*(treatedSums - controlPercent*treatedM + treatedPercent*controlM - controlSums)), axis = 0)
    overallDelta = uplift[-1]
    
    
    if graph:
        plt.plot([0,1], [0, overallDelta])
        plt.plot(np.linspace(0, 1, bins+1), uplift, label = 'model')
        plt.ylabel('Total % Increase Creatinine of SubPopulation')
        plt.xlabel('Fraction Targeted')
        plt.legend(loc = 'upper right')

    return np.trapz(uplift, dx = 1/bins) - 0.5*overallDelta



def classVariableTransform(Y, T, flip = False):
    ## expects numpy arrays, returns numpy array of same type
    Y = 1-Y if flip else Y
    Z = np.logical_not(np.logical_xor(Y,T))*1
    return Z

def keep_only_nums(data):
    dtype_set = {np.int8, np.int16, np.int32, np.float32, np.float64, np.bool}
    return data.loc[:,data.dtypes.apply(lambda x: any([issubclass(x.type, dtype) for dtype in dtype_set]))]

def remove_nan_cols(data):
    # only examine first row for nans
    return data.loc[:,data.iloc[0,:].apply(lambda x: not np.isnan(x))]

def clean(data):
    ## REMOVE BAD COLS
    data = remove_nan_cols(keep_only_nums(data))
    
    ## COMPRESS DTYPES
    # examine only the first 1000 rows
    new_dtypes = data.dtypes.copy()
    new_dtypes[data.iloc[:1000,:].apply(lambda x: len(np.unique(x)) <= 2, axis = 0)] = np.dtype('bool')
    new_dtypes[new_dtypes.apply(lambda x: issubclass(x.type, np.float64))]= np.dtype('float32')
    data = data.astype(new_dtypes.to_dict(), copy = True)
    return data
    

def buildIndexer(m, train_per = None, num_include = None):
    assert np.logical_xor(train_per is not None, num_include is not None)
    train_index = np.repeat(False, m)
    if train_per is not None:
        train_index[:int(round(m*train_per))] = True
    else:
        assert type(num_include) is int
        train_index[:num_include] = True
    return train_index
    
    
    
def standardize(data, train_index, exclude = []):
    vars_to_standardize = list(set(data.columns) - set(data.columns[data.dtypes == np.bool]) - set(exclude))
    scaler = StandardScaler()
    data.loc[train_index, vars_to_standardize] = scaler.fit_transform(data.loc[train_index, vars_to_standardize])
    data.loc[~train_index, vars_to_standardize] = scaler.transform(data.loc[~train_index, vars_to_standardize])
    return data



def one_hot(df, var_name):
    col = df[var_name].values.reshape(-1,1)
    all_values = np.unique(col)
    for i in range(len(all_values)):
        col[col == all_values[i]] = i
    oneHot = OneHotEncoder(dtype = 'int16')
    col_names = ['hispanic', 'black', 'white']
    col_one_hot = pd.DataFrame(oneHot.fit_transform(col).toarray(), index = df.index,
                           columns = col_names)
    df = pd.concat([df, col_one_hot], axis = 1)
    return df



def performance(data, bins = 10, graph = False, label = ' '):
    ## data should be 2-d numpy array with the following column order: uplift, target, treatment
    ################
    
    # sort data by descending uplift values
    sort_index = np.argsort(data[:,0])[::-1]
    data = data[sort_index,:]
    m = np.shape(data)[0]
    
    # find uplift values upon which to bin / add remainder patients to first bin by removing first cutoff
    uplift_cutoffs = data[np.arange(m)[::-int(float(m)/bins)][-2::-1],0]
    
    # separate into treated and control groups
    treated = data[data[:,2] == 1,:]
    control = data[data[:,2] == 0,:]
    
    # add cumulative outcomes and counts as columns 3 and 4, respectively
    treated = np.concatenate((treated, np.expand_dims(np.cumsum(treated[:,1]), axis = 1), 
                              np.expand_dims(np.arange(1, len(treated) + 1), axis = 1)), axis = 1)
    control = np.concatenate((control, np.expand_dims(np.cumsum(control[:,1]), axis = 1), 
                              np.expand_dims(np.arange(1, len(control) + 1), axis = 1)), axis = 1)
    
    
    # perform search for indices to measure auuc with
    treated_indices = np.zeros(bins, dtype = 'int')
    treated_indices[-1] = len(treated) - 1
    control_indices = np.zeros(bins, dtype = 'int')
    control_indices[-1] = len(control) - 1
    
    next_iter_treat, next_iter_control = 0, 0
    for counter in range(bins-1):
        for i in range(next_iter_treat, len(treated)):
            if treated[i,0] == uplift_cutoffs[counter]:
                treated_indices[counter] = i
                next_iter_treat = i + 1
                break
            elif treated[i,0] < uplift_cutoffs[counter]:
                treated_indices[counter] = i-1
                next_iter_treat = i + 1
                break
        for j in range(next_iter_control, len(control)):
            if control[j,0] == uplift_cutoffs[counter]:
                control_indices[counter] = j
                next_iter_control = j + 1
                break
            elif control[j,0] < uplift_cutoffs[counter]:
                control_indices[counter] = j-1
                next_iter_control = j + 1
                break
    
    # estimate heights at different points of the uplift curve
    # (y_treat - y_control_adjusted) + (y_treat_adjusted - y_control) all divided by total population size
    difference = 100*(treated[treated_indices,3] - \
                    control[control_indices,3]*treated[treated_indices,4]/control[control_indices,4]  + \
                 treated[treated_indices,3]*control[control_indices,4]/treated[treated_indices,4] - \
                    control[control_indices,3])/m
    
    # divide by bins to scale x-axis
    auuc_noAvg = np.trapz(difference, dx = 1./bins)
    
    # remove triangle with height of avg effect (final point on curve) and length-1 base to account for baseline uplift
    avg_effect = difference[-1]
    auuc = auuc_noAvg - 0.5*avg_effect
    
    if graph:
        plt.plot([0] + [(b+1)*100/bins for b in list(range(bins))], [0] + list(difference), label = label)
        plt.plot([0, 100], [0, avg_effect])
        plt.ylabel('% of Pop. Lives Saved')
        plt.xlabel('% of Pop. Targeted')
        plt.legend(loc = 'upper left')
    
    return(auuc)



def bars(data, U, bins = 5):
    #combine and sort data
    global targets, treatment
    IDs = data.index.values
    Y = 1 - data[targets].values[:,0]
    T = data[treatment].values
    sort_index = np.argsort(U)[::-1]
    data = pd.DataFrame({'Uplift': U, 'target': Y, 'treatment': T}, index = IDs).iloc[sort_index,:]
    
    # organize slices
    m = len(data)
    bin_size = int(np.floor(m/bins))
    slices = [slice(bin_size*i, bin_size*(i+1)) for i in range(bins-1)]
    slices += [slice(bin_size*(bins-1),m)]
    
    data_binned = [data.iloc[locations,:] for locations in slices]
    Y_t = [sum(data['target'][data['treatment']==1]) for data in data_binned]
    Y_c = [sum(data['target'][data['treatment']==0])for data in data_binned]
    N_t = [sum(data['treatment'] == 1) for data in data_binned]
    N_c = [sum(data['treatment'] == 0) for data in data_binned]
    P_t = [Y_t[i]/N_t[i] for i in range(bins)]
    P_c = [Y_c[i]/N_c[i] for i in range(bins)]
    
    fig, ax = plt.subplots()
    width = 0.3
    rects1 = ax.bar(np.arange(bins), P_t, width = width, color = 'b')
    rects2 = ax.bar(np.arange(bins) + width, P_c, width = width, color = 'r')

    ax.set_ylabel('% of Pop. Lives Saved')
    ax.set_xlabel('% of Pop. Targeted')
    ax.set_ylim([0.85, 0.98])
    ax.legend((rects1[0], rects2[0]), ('Treated', 'Control'))
    plt.show()

    




def descriptionUpdate(description, file):
    # remove first '\n' and grab model name
    description = description[1:]
    model_name = description[:description.index(':')]
    
    # read file
    f = open(file, 'r')
    contents = f.read()
    f.close()
    
    # collect all recorded model names
    splits = recursive_split(contents,[])
    if splits != []:
        splits = [splits[i][:splits[i].index(':')] for i in range(len(splits))]
        
    # append model description if name not taken
    if model_name in splits:
        raise ValueError('Model name is already used! Try another name.')
    f = open(file, 'a')
    f.write(description)
    f.close()

def recursive_split(contents, splits):
    index = contents.find('\n\n')
    if index == -1:
        return splits
    else:
        splits.append(contents[:index])
        contents = contents[index+2:]
        splits = recursive_split(contents, splits)
        return splits


def exponentialPropensity(input):
    x = np.abs(input - 0.5)
    # y = alpha*exp(-lambda*x) - beta; coeffients solved to pass through { (0,1), (0.5,0), (0.25,0.25) }
    alpha = 9./8
    beta = 1./8
    lambd = 2*np.log(9)
    y = alpha*np.exp(-lambd*x) - beta
    return y

    
#############################
#### 2 functons right here
searchName = lambda df, term: list(df.columns.values[[term in name for name in list(df.columns.values)]])
getWeights = lambda outcomes: 1./(2*np.mean(outcomes))

flatten = lambda l: [item for sublist in l for item in sublist]
###########################


def createTrainingMarkers(test_ids, all_ids):
    # test ids should be numpy matrix
    # all ids should be pandas index
    all_training_markers = pd.DataFrame(index = all_ids)
    test_ids = test_ids.astype('int')
    m, n = np.shape(test_ids)
    for i in range(n):
        test_ids_slice = test_ids[:,i]
        train_ids_slice = np.array(list(set(all_ids) - set(test_ids_slice)))
        m2 = len(train_ids_slice)
        marker = np.concatenate((np.ones(m2), np.zeros(m)))
        marker = pd.DataFrame(data = marker, index = np.concatenate((train_ids_slice, test_ids_slice)))
        all_training_markers[i + 1] = marker


def loadVar(var, grad = True):
    if type(var) in (pd.core.series.Series, pd.core.frame.DataFrame):
        var = var.values
    var = torch.from_numpy(var).cuda()
    return Variable(var, requires_grad = grad)




def createDict(file_name):
    var_dict = pd.read_csv(file_name, index_col = 0) == 1
    all_vars = var_dict.index.values
    var_dict = var_dict.to_dict('series')
    for name in var_dict:
        var_dict[name] = list(all_vars[var_dict[name]])
    return var_dict

def orderDict(var_dict, data):
    for name in var_dict:
        vars = var_dict[name]
        binary_vars = findCategorical(data[vars], cutoff = 2)
        var_dict[name] = binary_vars + list(set(vars) - set(binary_vars))
    print('Reordered.')
    return OrderedDict(sorted(var_dict.items(), key = lambda x: len(x[1])))

def findCategorical(data, cutoff = 2):
    data = data.fillna(value = 0)
    binary_bool = [len(np.unique(1*data[col].values)) <= cutoff for col in data.columns.values]
    return list(data.columns.values[binary_bool])

def loadDict(file, data):
    var_dict = createDict(file)
    var_dict['orders'] = list(data.loc[:,'ablationorder':'ventpcorder'].columns)
    var_dict = orderDict(var_dict, data = data)
    return var_dict


class BrownBoost(object):
    def __init__(self, X, Y, c):
        
        self.c = c                  # total time / target error rate
        self.s = c                  # amount of time remaining
        self.gamma = None
        self.counter = 0
        
        self.models = []
        self.alphas = []
        self.X = X
        self.Y = Y                  # true targets
        self.Y_adj = 2*Y - 1
        self.n = len(Y)
        self.r = np.zeros(self.n)        # margin of each sample
        self.w = np.ones(self.n)
        
        self.v = 1e-15              # small constant used to avoid degenerate cases
        self.step_size = 1e-5
        self.terminated = False
    
    def fit(self):
        while(not self.terminated):
            self.tick()
        
    def predict(self, X_new):
        l = len(self.models)
        m = len(X_new)
        preds = np.zeros((m,l))
        for i in range(l):
            alphas_norm = np.array(self.alphas)
            alphas_norm = alphas_norm/np.sum(alphas_norm)
            preds[:,i] = alphas_norm[i]*self.models[i].predict_proba(X_new)[:,1]
        preds = np.sum(preds, axis = 1)
        return preds
    
    def tick(self):
        print('Iteration: ' + str(self.counter) + '\t Time Left: ' + str(self.s))
        self.w = self.calculateWeights()
        
        # learn weak hypothesis
        self.models += [LR(C = 1)]
        self.models[self.counter].fit(self.X, self.Y, sample_weight = self.w)
        # get predictions / parameters
        h = 2*self.models[self.counter].predict_proba(self.X)[:,1] - 1
        alpha, t = self.solveDifferential(h)
        self.alphas += [alpha]
        ## update stuff
        self.updateMargin(alpha, h)
        self.updateTime(t)
        self.counter += 1
        self.terminated = True if self.s <= 0 else False
            
    def calculateWeights(self):
        w =  np.exp(-np.power(self.r + self.s, 2)/self.c)
        return self.n*w/np.sum(w)
        
    
    def calculateGamma(self, alpha, t, h):
        hY = h*self.Y_adj
        gamma_pre =  np.exp(-np.power(self.r + alpha*hY + self.s - t, 2)/self.c)
        gamma = np.sum(gamma_pre*hY)/np.sum(gamma_pre)
        return gamma
    
    def solveDifferential(self, h):
        alpha, t = 0, 0
        gamma = self.calculateGamma(alpha, t, h)
        while gamma > self.v and t < self.s:
            t += self.step_size
            alpha += self.step_size/gamma
            gamma = self.calculateGamma(alpha, t, h)
        return alpha, t
    
    def updateMargin(self, alpha, h):
        self.r = self.r + alpha*h*self.Y_adj
        
    def updateTime(self, t):
        self.s -= t
        if self.s < 0:
            self.terminated = True    
        



class DataLoader(object):
    def __init__(self, train_path, holdOut_path):
        self.train_path = train_path
        self.holdOut_path = holdOut_path
        
    def load(self, mode):
        assert mode in ['train', 'test']
        store_file_name = self.train_path if mode is 'train' else self.holdOut_path
        store = open(store_file_name, 'rb')
        data = pickle.load(store)
        store.close()
        return data




#####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#####   
###############     BEGINNING of:                  UPLIFT DATA CONTAINER CLASS                              ###############

class UpliftDataContainer(object):
    '''
    data:               pandas.DataFrame
    info:               dict = {'ID': 'id_string', 'predictors': ['predictor0', ...], 
                                'targets': ['target0', ...], 'treatment' = 'assignment_string'}
    splitProportions:   list of floats in (0,1) = [train, val, test] (fractions of dataset you want in each group)
                            if replacement is True, training and validation sets will be calculated differently than this
    batchSize:          int (number of instances fed during iterator calls)
    temporal:           bool (use the temporal test split or a randomized one)
    replacement:        bool (sample train splits with/without replacement)
    verbose:            bool (whether the algorithm should print information about what it's doing)
    '''
    def __init__(self, data, info, splitProportions = [0.6, 0.2, 0.2], batchSize = 64, 
                         temporal = False, replacement = False, verbose = False):
        super().__init__()
        assert type(data) is pd.core.frame.DataFrame and type(info) is dict
        assert len(splitProportions) == 3 and sum(splitProportions) == 1 and all([proportion > 0 for proportion in splitProportions])
        assert type(batchSize) is int and batchSize >= 1
        assert all([type(x) is bool for x in (temporal, replacement, verbose)])
        assert all([x in info.keys() for x in ('predictors', 'targets', 'ID', 'treatment')])
        self.treatedRaw, self.treatedTrainVal, self.treatedTrain, self.treatedVal, self.treatedTest = (None for _ in range(5))
        self.controlRaw, self.controlTrainVal, self.controlTrain, self.controlVal, self.controlTest = (None for _ in range(5))
        self.imputer, self.scaler = None, None

        ## record important initialization information
        self.predictors, self.targets = info['predictors'], info['targets']
        self.id, self.treatment =  info['ID'], info['treatment']
        self.splitProportions = splitProportions
        self.batchSize = batchSize
        self.replacement = replacement
        self.verbose = verbose
        
        
        ## clean data and sort into appropriate groups.
        # lambda function below creates indexers for every combination of people that fall
        # inside / outside the categories contained in the list 'groups' (4 combinations)
        self.dataRaw = self.__clean(data.set_index(self.id, drop = True, inplace = False))
        self.treated = self.dataRaw[self.dataRaw[self.treatment] == 1]
        self.control = self.dataRaw[self.dataRaw[self.treatment] == 0]
        groups = ['icuatalert', 'surgical']
        createGroupIndexers = lambda population: [np.logical_and(population[groups[0]] == 0, population[groups[1]] == 0),
                                            np.logical_and(population[groups[0]] == 1, population[groups[1]] == 0),
                                            np.logical_and(population[groups[0]] == 0, population[groups[1]] == 1),
                                            np.logical_and(population[groups[0]] == 1, population[groups[1]] == 1)]
        groupIndexersTreated = createGroupIndexers(self.treated)
        groupIndexersControl = createGroupIndexers(self.control)
        self.treatedRaw = [self.treated[index] for index in groupIndexersTreated]
        self.controlRaw = [self.control[index] for index in groupIndexersControl]
        
        ## create appropriate data splits with imputation and scaling
        self.reboot(temporal = temporal)
        
    
    
            
    ###############     END of:                            INITIALIZATION                                     ###############
    #######========-------------------------------------------------------------------------------------------========#######
    ###############     BEGINNING of:                       USER METHODS                                      ###############
    
    
    def setVerbose(self, verbose):
        self.verbose = verbose
    def setBatchSize(self, batchSize):
        self.batchSize = batchSize
    def setSplitProportions(self, splitProportions):
        self.splitProportions = splitProportions
    def setPredictors(self, predictors):
        self.predictors = predictors
        self.__set_predictor_groups()
    def setTargets(self, targets):
        self.targets = targets
    def setReplacement(self, replacement):
        self.replacement = replacement
    
    
    ## run this function when you want to create a new held-out test set (essentailly start from fresh)
    def reboot(self, temporal = False, verbose = None):
        verbose = self.verbose if verbose is None else verbose
        trainValProportion = self.splitProportions[0] + self.splitProportions[1]
        indexersTreated = [self.__buildIndexer(len(subGroup), trainValProportion) for subGroup in self.treatedRaw]
        indexersControl = [self.__buildIndexer(len(subGroup), trainValProportion) for subGroup in self.controlRaw]
        cOutcomerate = lambda data: sum(data.loc[data[self.treatment] == 0, self.targets])/sum(data[self.treatment] == 0)
        
        reject = True
        while reject:
            ## randomly shuffle or organize temporally ALL the data
            if temporal:
                self.treatedRaw = [subGroup.sort_values(by = 'timesec') for subGroup in self.treatedRaw]
                self.controlRaw = [subGroup.sort_values(by = 'timesec') for subGroup in self.controlRaw]
                reject = False
            else:
                self.treatedRaw = [subGroup.sample(frac = 1., replace = False) for subGroup in self.treatedRaw]
                self.controlRaw = [subGroup.sample(frac = 1., replace = False) for subGroup in self.controlRaw]
                    
            
            ## make (Train + Val) / Test  Split
            # copy data to ensure imputations, scaling, etc... don't effect raw data -
            # this is necessary for the sake of future reboots

            self.treatedTrainVal = [subGroup[indexer].copy() for indexer, subGroup in zip(indexersTreated, self.treatedRaw)]
            self.controlTrainVal = [subGroup[indexer].copy() for indexer, subGroup in zip(indexersControl, self.controlRaw)]
            self.treatedTest = [subGroup[~indexer].copy() for indexer, subGroup in zip(indexersTreated, self.treatedRaw)]
            self.controlTest = [subGroup[~indexer].copy() for indexer, subGroup in zip(indexersControl, self.controlRaw)]
#            if not temporal and (abs(cOutcomerate(self.controlTest[0]) - 0.1423) < 0.05 and    \
#                   abs(cOutcomerate(self.controlTest[1]) - 0.3376) < 0.05 and    \
#                   abs(cOutcomerate(self.controlTest[2]) - 0.1293) < 0.05 and    \
#                   abs(cOutcomerate(self.controlTest[3]) - 0.2562) < 0.05):
            reject = False

            
        if verbose:
            print('Splitting data into (Train + Validation) and Test sets: successful')
        
        ## impute values for each subGroup separately, but combining treatment and control
        self.__imputeFit([pd.concat([t, c], axis = 0)[self.predictors].values               \
                                                          for t,c in zip(self.treatedTrainVal, self.controlTrainVal)])
        for i in range(4):
            self.treatedTrainVal[i][self.predictors] = self.__imputeTransform(self.treatedTrainVal[i][self.predictors].values, i)
            self.controlTrainVal[i][self.predictors] = self.__imputeTransform(self.controlTrainVal[i][self.predictors].values, i)
            self.treatedTest[i][self.predictors] = self.__imputeTransform(self.treatedTest[i][self.predictors].values, i)
            self.controlTest[i][self.predictors] = self.__imputeTransform(self.controlTest[i][self.predictors].values, i)
        if verbose:
            print('Imputing missing data: successful')
        
        ## standardize all subGroups combining treatment and control together
        self.__scaleFit(pd.concat(self.treatedTrainVal + self.controlTrainVal, axis = 0)[self.predictors].values)
        for i in range(4):
            self.treatedTrainVal[i][self.predictors] = self.__scaleTransform(self.treatedTrainVal[i][self.predictors].values)
            self.controlTrainVal[i][self.predictors] = self.__scaleTransform(self.controlTrainVal[i][self.predictors].values)
            self.treatedTest[i][self.predictors] = self.__scaleTransform(self.treatedTest[i][self.predictors].values)
            self.controlTest[i][self.predictors] = self.__scaleTransform(self.controlTest[i][self.predictors].values)
        if verbose:
            print('Scaling all features: successful')
        
        self.refresh(replacement = self.replacement, verbose = verbose)
            
        

    
    def refresh(self,replacement = None, verbose = None):
        replacement = self.replacement if replacement is None else replacement
        verbose = self.verbose if verbose is None else verbose
        
        ## select the indices for the training group
        if replacement:
            bagIndices = lambda x: np.random.choice(np.arange(len(x)), size = len(x))
            trainIndicesTreated = [bagIndices(subGroup) for subGroup in self.treatedTrainVal]
            trainIndicesControl = [bagIndices(subGroup) for subGroup in self.controlTrainVal]
        else:
            trainProportion = self.splitProportions[0] / (self.splitProportions[0] + self.splitProportions[1])
            trainIndicesTreated = [np.arange(len(subGroup))[self.__buildIndexer(len(subGroup), trainProportion)]        \
                                                                         for subGroup in self.treatedTrainVal]
            trainIndicesControl = [np.arange(len(subGroup))[self.__buildIndexer(len(subGroup), trainProportion)]        \
                                                                         for subGroup in self.controlTrainVal]
        
        ## let the validation set be the indices not included in training
        valIndicesTreated = [np.setdiff1d(np.arange(len(subGroup)), index)          \
                                             for index, subGroup in zip(trainIndicesTreated, self.treatedTrainVal)]
        valIndicesControl = [np.setdiff1d(np.arange(len(subGroup)), index)          \
                                             for index, subGroup in zip(trainIndicesControl, self.controlTrainVal)]
        
        ## subset train and val, accordingly
        self.treatedTrain = [subGroup.iloc[index,:] for index, subGroup in zip(trainIndicesTreated, self.treatedTrainVal)]
        self.controlTrain = [subGroup.iloc[index,:] for index, subGroup in zip(trainIndicesControl, self.controlTrainVal)]
        self.treatedVal = [subGroup.iloc[index,:] for index, subGroup in zip(valIndicesTreated, self.treatedTrainVal)]
        self.controlVal = [subGroup.iloc[index,:] for index, subGroup in zip(valIndicesControl, self.controlTrainVal)]
        self.treatedForGenerator = pd.concat(self.treatedTrain, axis = 0)
        self.controlForGenerator = pd.concat(self.controlTrain, axis = 0)
        if verbose:
            print('Splitting data into Train and Validation sets: successful')
            
    
    
    ## returns length of a dataset
    # __len__ won't allow multiple arguments
    # join determines if treated and control are combined
    def size(self, datasetName, join = True):
        datasetName = self.__cleanName(datasetName)
        assert datasetName in ['train', 'val', 'test', 'all']
        if datasetName == 'train':
            treated, control = self.treatedTrain, self.controlTrain
        elif datasetName == 'val':
            treated, control = self.treatedVal, self.controlVal
        elif datasetName == 'test':
            treated, control = self.treatedTest, self.controlTest
        else:
            treated, control = self.treatedRaw, self.controlRaw
        Nt, Nc = sum([len(subGroup) for subGroup in treated]), sum([len(subGroup) for subGroup in control])
        return Nt+Nc if join else (Nt, Nc)
    
            
    ###############     END of:                              USER METHODS                                     ###############
    #######========-------------------------------------------------------------------------------------------========#######
    ###############     BEGINNING of:                     EXCLUSIVE CLASS METHODS                             ###############

    
    ## performs imputations within each subGroup
    # data should be list of subGroups with each subGroup being a numpy array
    def __imputeFit(self, data):
        k = len(data)
        self.imputer = [Imputer(strategy = 'mean', copy = True) for i in range(k)]
        for i in range(k):
            self.imputer[i].fit(data[i])
    ## data here should instead be a numpy array and index indicates which imputer to use
    def __imputeTransform(self, data, index):
        assert index in range(len(self.imputer))
        return self.imputer[index].transform(data)

    ## data should be a single numpy array with all training + val data
    def __scaleFit(self, data):
        self.scaler = StandardScaler()
        self.scaler.fit(data)
    def __scaleTransform(self, data):
        return self.scaler.transform(data)
        
    def __buildIndexer(self, m, percentInclude = None, numInclude = None):
        assert np.logical_xor(percentInclude is None, numInclude is None)
        index = np.repeat(False, m)
        if percentInclude is not None:
            index[:int(round(m*percentInclude))] = True
        else:
            assert type(numInclude) is int
            index[:numInclude] = True
        return index
    

    def __clean(self, data):
        ## removes pure nan columns and strings
        dtype_set = {np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, np.bool}
        data = data.loc[:,data.dtypes.apply(lambda x: any([issubclass(x.type, dtype) for dtype in dtype_set]))]
        
        ## changes all remaining data types to float32 for the sake of future computation
        new_dtypes = data.dtypes.copy()
        new_dtypes[np.array([True]*len(new_dtypes))] = np.dtype('float32')
        return data.astype(new_dtypes.to_dict())


    def __cleanName(self, datasetName):
        datasetName = datasetName.lower().strip()
        datasetName = 'val' if datasetName == 'validation' else datasetName
        return datasetName
    
    
    ###############     END of:                          EXCLUSIVE CLASS METHODS                              ###############
    #######========-------------------------------------------------------------------------------------------========#######
    ###############    BEGINNING of:                      PYTHON SPECIAL METHODS                              ###############


    
    ## generator iterates through treated and control training groups, assuming both their subgroups havebeen shuffled
    # join determines whether one pandas df is returned (per treatment) or a list containing 4 dfs from each subGroup 
    def __iter__(self):
        self.treatedForGenerator = self.treatedForGenerator.sample(frac = 1., replace = False, axis = 0)
        self.controlForGenerator = self.controlForGenerator.sample(frac = 1., replace = False, axis = 0)
        self.Nt, self.Nc = self.size('train', join = False)
        self.terminate = False
        self.index, self.end = 0, self.batchSize
        return self
    def __next__(self):
        if self.terminate:
            raise StopIteration()
        
        batchTrain = self.treatedForGenerator.iloc[self.index:self.end,:]
        batchControl = self.controlForGenerator.iloc[self.index:self.end,:]
        ## setup for next potential round.
        # if either group won't have enough for a full batch next round:
        # then add the remaining samples to the batch and terminate the generator after current return         
        if self.end + self.batchSize >= self.Nt or self.end + self.batchSize >= self.Nc:
            batchTrain = self.treatedForGenerator.iloc[self.index:,:]
            batchControl = self.controlForGenerator.iloc[self.index:,:]
            self.terminate = True
        self.index = self.end
        self.end += self.batchSize     
        return batchTrain, batchControl
    
    
    ## returns a dataset
    # join determines if treated and control are combined
    def __call__(self, datasetName, join = True):
        datasetName = self.__cleanName(datasetName)
        assert datasetName in ['train', 'val', 'test']
        if datasetName == 'train':
            treated, control = self.treatedTrain, self.controlTrain
        elif datasetName == 'val':
            treated, control = self.treatedVal, self.controlVal
        else:
            treated, control = self.treatedTest, self.controlTest
        treated, control = pd.concat(treated, axis = 0), pd.concat(control, axis = 0)
        return pd.concat([treated, control], axis = 0) if join else (treated, control)
            
        
        
    
###############     END of:                  UPLIFT DATA CONTAINER CLASS                                  ###############
#####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#####



class SkewCorrection(object):
    def __init__(self, p = 0.05, copy = True):
        super().__init__()
        self.p = p
        self.copy = copy
        self.fitted = False
        
        
    def fit(self, data):
        data = data.copy() if self.copy else data
        n = np.shape(data)[1]
        
        skewed_bool = [skewtest(data[:,i])[1] < self.p for i in range(n)]
        self.indices = np.array(list(range(n)))[skewed_bool]
        self.indices = list(self.indices)
        g = len(self.indices)
        self.lambdas, self.minVals, self.nonNegative = [None]*g , [None]*g , [None]*g   

        for i, index in enumerate(self.indices):
            self.__single_boxcox_fit(data[:,index], i)
        self.fitted = True
        
    def transform(self, data):
        assert self.fitted
        data = data.copy() if self.copy else data
        for i, index in enumerate(self.indices):
            data[:,index] = self.__single_boxcox_transform(data[:, index], i)
        return data
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
        
    def __single_boxcox_fit(self, col, index):
        
        self.minVals[index] = np.min(col)
        self.nonNegative[index] = True if self.minVals[index] >= 0 else False
        col = self.__minTrans(self.nonNegative[index], col, self.minVals[index])
        
        _, lmbda = boxcox(col, lmbda = None)
        self.lambdas[index] = lmbda
        
    def __single_boxcox_transform(self, col, index):
        col = self.__minTrans(self.nonNegative[index], col, self.minVals[index])
        result = boxcox(col, lmbda = self.lambdas[index])
        return result
        
    def __minTrans(self, nonNegative, col, minVal):
        if nonNegative == True:
            col = col - minVal + 20
        else:
            col = col - 7*minVal + 20
        return col


   