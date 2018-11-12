#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 13:54:28 2017

@author: aditya
"""
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoLars
from sklearn.preprocessing import Imputer, StandardScaler
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn import svm
import itertools
from pyDOE import lhs

trainIndex = np.repeat(False, len(df))
trainIndex[:int(round(len(df)*0.5))] = True
valIndex = np.repeat(False, len(df))

action = 'assignment'
target = 'creatoutcome7percent'

predictorBank = [ 'nsaidcategory','chloride', 'glucose', 'pressorcategory', 'hematocrit', 'mch', 'lactate', 'phosphorus',
             'chf', 'basophilpercent',
             'lymphpercent', 'leukocyteabs', 'neutrophilpercent', 'monopercent', 'monoabs', 'eosinophilabs', 'potassium']

predictorBase = 'cratio0 cratio1 mcv mchc icuatalert c0value c1value bicarbonate bun age malegender hemoglobin wbc plateletcount sodium surgical'.split(' ')
predictorsExtra = ['uaprotein', 'timesec', 'redcelldistribution', 'alkphos', 'cdeltapercent', 'cslope',
                  'aaorno', 'pt', 'eospercent', 'paralyticcategory', 'neutrophilabs', 'basosabs', 'mch','uaspecgrav',
                  'bilitotal', 'magnesium', 'missing', 'orders']
predictorsExtra = predictorBank + predictorsExtra


predictors = predictorBase.copy()
predictors += ['cratio0', 'cratio1']


bestAUUC = 1000
bestPredictors = []
for j in range(len(predictorBank)):

    predictors = predictorBase + [predictorBank[j]]
    
    
    iterations = 50
    auucs = np.zeros(iterations)
    auucs2 = np.zeros(iterations)
    for i in range(iterations):
        df = df.sample(frac = 1.)
        #df = df.sort_values(by = 'timesec')
        dfTrain = df[trainIndex]
        dfTest = df[testIndex]
        dfVal = df[valIndex]
        imputer = Imputer()
        scaler = StandardScaler()
        xTrain = imputer.fit_transform(dfTrain[predictorBase + predictorsExtra].values)
        xVal = imputer.transform(dfVal[predictorBase + predictorsExtra].values)
        xTest = imputer.transform(dfTest[predictorBase + predictorsExtra].values)
        xTrain = pd.DataFrame(scaler.fit_transform(xTrain), columns = predictorBase + predictorsExtra)
        xVal = pd.DataFrame(scaler.transform(xVal), columns = predictorBase + predictorsExtra)
        xTest = pd.DataFrame(scaler.transform(xTest), columns = predictorBase + predictorsExtra)
#        xTrainPoly = pieceFeature(dfTrain['surgical'].values, pieceFeature(dfTrain['icuatalert'].values, xTrain))
#        xTestPoly = pieceFeature(dfTest['surgical'].values, pieceFeature(dfTest['icuatalert'].values, xTest))
    
        
        yTrain, yVal, yTest = dfTrain[target].values, dfVal[target].values, dfTest[target].values
        aTrain, aVal, aTest = dfTrain[action].values == 1, dfVal[action].values == 1, dfTest[action].values == 1
        
        
#        modelAction = LogisticRegression(C=1)
#        modelAction.fit(xTrainPoly, aTrain)
#        aEst = modelAction.predict_proba(xTrainPoly)
#        weightsT = 1/aEst[aTrain,1]
#        weightsC = 1/aEst[~aTrain,0]
        
        predictorsCurrent = predictorBase.copy()
        searching = True
        bestAUUC = 1e8
        while searching:
            bestSubAUUC = 1e8
            for k in range(len(predictorsExtra)):
                predictors = predictorsCurrent + [predictorsExtra[k]]
                
                modelC = LinearRegression()
                modelC.fit(xTrain.loc[~aTrain, predictors].values, yTrain[~aTrain])#, sample_weight = weightsC)
                p = modelC.predict(xTrain.loc[aTrain, predictors].values)
                error = -(yTrain[aTrain] - p)
        
                
                modelT = LinearRegression()
                modelT.fit(xTrain.loc[aTrain, predictors].values, yTrain[aTrain])#, sample_weight = weightsT)
                p2 = modelT.predict(xTrain.loc[~aTrain, predictors].values)
                error2 = (yTrain[~aTrain] - p2)
                #error2[yTrain[~aTrain] <= 0] = 0
                #sns.distplot(error)
                
                modelUplift = LinearRegression()
                modelUplift.fit(np.concatenate((xTrain.loc[aTrain, predictors], xTrain.loc[~aTrain, predictors]), axis = 0), 
                                np.concatenate((error, error2), axis = 0))
                #modelUplift.fit(xTrain[aTrain], error)
                uVal = modelUplift.predict(xVal[predictors].values)
        #        uTest = modelProg.predict(xTest) - modelProg2.predict(xTest)
                uCheck, yCheck, aCheck = uVal, yVal, aVal
                #uCheck, yCheck, aCheck = uTrain, yTrain, aTrain
                
                auuc = AUUC(uCheck, yCheck, aCheck, graph = False)
                if auuc < bestSubAUUC:
                    currentFeature = predictorsExtra[k]
                    bestSubAUUC = auuc
            if bestSubAUUC < bestAUUC:
                bestAUUC = bestSubAUUC
                predictorsCurrent += [currentFeature]
            else:
                break
        
        predictors = predictorsCurrent
        modelC = LinearRegression()
        modelC.fit(xTrain.loc[~aTrain, predictors].values, yTrain[~aTrain])#, sample_weight = weightsC)
        p = modelC.predict(xTrain.loc[aTrain, predictors].values)
        error = -(yTrain[aTrain] - p)

        
        modelT = LinearRegression()
        modelT.fit(xTrain.loc[aTrain, predictors].values, yTrain[aTrain])#, sample_weight = weightsT)
        p2 = modelT.predict(xTrain.loc[~aTrain, predictors].values)
        error2 = (yTrain[~aTrain] - p2)
        #error2[yTrain[~aTrain] <= 0] = 0
        #sns.distplot(error)
        
        modelUplift = LinearRegression()
        modelUplift.fit(np.concatenate((xTrain.loc[aTrain, predictors], xTrain.loc[~aTrain, predictors]), axis = 0), 
                        np.concatenate((error, error2), axis = 0))
        #modelUplift.fit(xTrain[aTrain], error)
        #uTrain = modelUplift.predict(xTrain[predictors].values)
        uTest = modelUplift.predict(xTest[predictors].values)
#        uTest = modelProg.predict(xTest) - modelProg2.predict(xTest)
        uCheck, yCheck, aCheck = uTest, yTest, aTest
        #uCheck, yCheck, aCheck = uTrain, yTrain, aTrain
        
        auucs[i] = AUUC(uCheck, yCheck, aCheck, graph = False)
        
                
#        uTest = modelC.predict(xTest) - modelT.predict(xTest)
#        uCheck, yCheck, aCheck = uTest, yTest, aTest        
#        auucs2[i] = AUUC(uCheck, yCheck, aCheck, graph = False)



    if np.mean(auucs) < bestAUUC:
        bestAUUC = np.mean(auucs)
        bestPredictors = predictors





###########################################33
######3 SINGLE NETWORK


selu = lambda input: 1.050700987355480493419335*F.elu(input, alpha = 1.673263242354377284817043)

#1.67653251702*
swish = lambda input: 1.67653251702*input*F.sigmoid(input)

class VariationalDropoutLayer(nn.Module):
    def __init__(self, inputSize, outputSize, nonlinearity = selu):
        super().__init__()
        self.linear = nn.Linear(inputSize, outputSize)
        init.kaiming_normal(self.linear.weight)
        init.constant(self.linear.bias, 0)
        
        self.logSigma2 = Parameter(torch.ones(self.linear.weight.size()), requires_grad = True)
        init.constant(self.logSigma2, -10)
        self.nl = nonlinearity
        self.TINY = 1e-8
        
        
    def calcKL(self, logAlpha):
        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695
        C = -k1
        negKL = k1*F.sigmoid(k2 + k3*logAlpha) - 0.5*torch.log(1 + 1./torch.exp(logAlpha)) + C
        return -negKL.sum()
    

    def forward(self, input, epoch = 1000):
        penalty = Variable(torch.zeros(1).cuda())
        W = Variable(self.linear.weight.data.clone(), requires_grad = False)
        W2 = W*W
        logAlpha = torch.clamp(self.logSigma2 - torch.log(W2 + self.TINY), -5, 5)
        
        if self.training and epoch > 100:
            penalty = self.calcKL(logAlpha)
            
            # testing but wanting to sample from distribution over weights
            # or training with KL penalty on
            mu = self.linear.forward(input)
            s = torch.sqrt(torch.mm(input*input, torch.t(torch.exp(logAlpha)*W2)))
                
            epsilon = Variable(torch.normal(torch.zeros(mu.size()).cuda(), std = 1))
            b = mu + epsilon*s
        else:
            
            # testing but wanting to make point estimates on weights (for validation set evaluation)
            # or training with KL penalty off
            b = self.linear.forward(input)

        return self.nl(b), penalty



class Classifier(nn.Module):
    def __init__(self, inputSize, hiddenSize, finalSize):
        super().__init__()
        #self.bn = nn.BatchNorm1d(inputSize)
        self.linearPre = nn.Linear(inputSize, hiddenSize*2)
        init.kaiming_normal(self.linearPre.weight)
        init.constant(self.linearPre.bias, 0)
        self.linearT = nn.Linear(inputSize, finalSize)
        init.kaiming_normal(self.linearT.weight)
        init.constant(self.linearT.bias, 0)
        self.linearC = nn.Linear(inputSize, finalSize)
        init.kaiming_normal(self.linearC.weight)
        init.constant(self.linearC.bias, 0)
        self.hiddenSize = hiddenSize
    
        
    def forward(self, input):
        #hPre = swish(self.linearPre(input))
        #hT = hPre[:,:self.hiddenSize]
        #hC = hPre[:,self.hiddenSize:]
        yT, yC = self.linearT(input), self.linearT(input)
        return yT[:,0], yC[:,0]


class ResBlock(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super().__init__()
        self.linear1 = VariationalDropoutLayer(inputSize, hiddenSize)
        #init.kaiming_normal(self.linear1.weight)
        #init.constant(self.linear1.bias, 0)
        self.linear2 = VariationalDropoutLayer(hiddenSize, inputSize)
        #init.kaiming_normal(self.linear2.weight)
        #init.constant(self.linear2.bias, 0)
        
    def forward(self, input):
        h, KL1 = self.linear1(swish(input))
        epsilon, KL2 = self.linear2(swish(h))
        z = input + epsilon
        KL = KL1 + KL2
        return z, KL

class Encoder(nn.Module):
    def __init__(self, inputSize, hiddenSize, finalSize):
        super().__init__()
        #self.linear1 = VariationalDropoutLayer(inputSize, finalSize)
        self.linear1 = VariationalDropoutLayer(inputSize, hiddenSize)
        #init.kaiming_normal(self.linear1.weight)
        #init.constant(self.linear1.bias, 0)
        #self.linear2 = VariationalDropoutLayer(hiddenSize*2, hiddenSize)
        #self.linear2 = VariationalDropoutLayer(hiddenSize, finalSize)
        self.linear2 = ResBlock(hiddenSize, hiddenSize*2)
        self.linear3 = ResBlock(hiddenSize, hiddenSize*2)
        self.linear4 = VariationalDropoutLayer(hiddenSize, finalSize)
        #init.kaiming_normal(self.linear4.weight)
        #init.constant(self.linear4.bias, 0)
        
        self.drop = nn.Dropout(p = 0.2)
        self.drop2 = nn.Dropout(p = 0.2)
        
    def forward(self, input, epoch = 1000):
        h1, KL1 = self.linear1(input)
        
        h2, KL2 = self.linear2(swish(h1))
        h3, KL3 = self.linear3(h2)
        h4, KL4 = self.linear4(h3)
        #std = self.linear3(h1)
        #KL = torch.mean(torch.sum(mean*mean + std*std - torch.log(std*std) - 1, dim = 1))
#        if self.training:
#            epsilon = Variable(torch.normal(torch.zeros(mean.size()).cuda(), std = 1))
#            h2 = mean + epsilon*std
#        else:
#            h2 = mean
        #h3, KL3 = self.linear3(h2)
        return F.leaky_relu(h4), KL1 + KL2 + KL3 + KL4


class Discriminator(nn.Module):
    def __init__(self, inputSize, hiddenSize, finalSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, finalSize)
        self.bn = nn.BatchNorm1d(hiddenSize)
    
    def forward(self, input):
        h = F.relu(self.bn(self.linear1(input)))
        y = F.sigmoid(self.linear2(h))
        return y[:,0]
    
class Delta(nn.Module):
    def __init__(self, inputSize, hiddenSize, finalSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, finalSize)
        init.kaiming_normal(self.linear1.weight)
        init.constant(self.linear1.bias, 0)
        #self.linear2 = nn.Linear(hiddenSize, finalSize)

    
    def forward(self, input):
        h = self.linear1(input)
        #y = self.linear2(h)
        return h[:,0]
    

class AE(nn.Module):
    def __init__(self, inputSize, hiddenSize, finalSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        init.kaiming_normal(self.linear1.weight)
        init.constant(self.linear1.bias, 0)
        self.linear2 = nn.Linear(hiddenSize, finalSize)
        init.kaiming_normal(self.linear2.weight)
        init.constant(self.linear2.bias, 0)

    
    def forward(self, input):
        h = F.elu(self.linear1(input))
        y = self.linear2(h)
        return y




def assymetricalLoss(prediction, target):
    error = target - prediction
    predBelow = F.relu(error)
    predAbove = F.relu(-error)
    loss = torch.mean(torch.pow(predBelow, 2) + predAbove)
    return loss


trainIndex = np.repeat(False, len(df))
valIndex = np.repeat(False, len(df))
testIndex = np.repeat(False, len(df))

cutTrain = int(round(len(df)*0.5))
cutVal = int(round(len(df)*0.7))


trainIndex[:cutTrain] = True
valIndex[cutTrain:cutVal] = True
testIndex[cutVal:] = True




hyperParams = {'lambda': [2e-2],
               'decay': [1e-4]}

paramNames = sorted(hyperParams)
combinations = [dict(zip(paramNames, prod)) for prod in itertools.product(*(hyperParams[paramName] for paramName in paramNames))]



counter = 0
iterations = 50
allAUUCs = np.zeros(iterations)

while counter < iterations:
    X, Y, A = refresh(temporal = True)
    xTrainVal, xTest = X
    yTrainVal, yTest = Y
    aTrainVal, aTest = A
    


    cvIters = 2
    predictorList = [None for i in range(cvIters)]
    hyperparamsBest = [None for i in range(cvIters)]
    for j in range(cvIters):
        print(j)
        shuffleIndices = np.random.permutation(range(len(yTrainVal)))
        xTrainVal, yTrainVal, aTrainVal = xTrainVal.iloc[shuffleIndices,:], yTrainVal[shuffleIndices], aTrainVal[shuffleIndices]
        xTrain, xVal = xTrainVal.iloc[trainIndex,:], xTrainVal.iloc[valIndex,:]
        yTrain, yVal = yTrainVal[trainIndex], yTrainVal[valIndex]
        aTrain, aVal = aTrainVal[trainIndex], aTrainVal[valIndex]
        
        predictorList[j] = featureSelect(XLearner, xTrain, xVal, yTrain, yVal, aTrain, aVal)
        #predictorList[j] = predictorBase + predictorBank
        
        xVal = loadVar(xVal[predictorList[j]].values.astype('float32'))
        valAUUC, auucBest = 9, 9
        for combo in combinations:
            epochs = 400
            encodingSize = 6
            encoder = Encoder(len(predictorList[j]), 12, encodingSize).cuda()
            model = Classifier(encodingSize, int(encodingSize/2), 1).cuda()
            #discriminator = Discriminator(encodingSize, 8, 1).cuda()
            G = Delta(encodingSize, 4, 1).cuda()
            #autoencoder = AE(encodingSize, 32, len(predictors)).cuda()
            
            
            opt = Adam(model.parameters(), lr = 1e-3)
            optEnc = Adam(encoder.parameters(), lr = 1e-3, weight_decay = combo['decay'])
            #optDisc = Adam(discriminator.parameters(), lr = 1e-3)
            optG = Adam(G.parameters(), lr = 1e-3)
            #optAE = Adam(autoencoder.parameters(), lr = 1e-3)
            
            
            dataLoader = DataLoader(dataset = TensorDataset(torch.from_numpy(xTrain[predictorList[j]].values.astype('float32')).cuda(), 
                                                            torch.from_numpy(np.stack((yTrain, aTrain), axis = 1).astype('float32')).cuda()),
                                    batch_size = 100, shuffle = True)
            
        
            
            
            epoch = -1
            n = int(sum(trainIndex))
            encoder.train()
            model.train()
            G.train()
            #discriminator.train()
            while epoch < epochs:
                epoch += 1
                runningLossF, runningLossC, runningLossKL, runningLossEnc, runningLossAE = 0, 0, 0, 0, 0
                
                if epoch == 300:
                    opt = Adam(model.parameters(), lr = 5e-3)
                    optG = Adam(G.parameters(), lr = 5e-3)
                    optEnc = Adam(encoder.parameters(), lr = 5e-3, weight_decay = combo['decay'])
                    
        
                
                for batchCounter, batch in enumerate(dataLoader):
                    Xtrain, Otrain = batch
                    Xtrain, Ytrain, Atrain = Variable(Xtrain), Variable(Otrain[:,0]), Otrain[:,1].byte()
                    Ytrain_treated, Ytrain_control = Ytrain[Atrain], Ytrain[1-Atrain]
                    
                    ## model stuff
                    h, KL = encoder(Xtrain)
                    #h2 = torch.cat((h, Xtrain[:,:len(predictorBase)]), dim = 1)
                    yT, yC = model(h)
                    g = G(h)
                    
                    
                    ## losses
                    yT_treated, yC_treated = yT[Atrain], yC[Atrain]
                    yT_control, yC_control = yT[1-Atrain], yC[1-Atrain]
                    
                    factualLoss = 0.5*torch.mean(torch.pow(yT_treated - Ytrain_treated, 2))
                    factualLoss += 0.5*torch.mean(torch.pow(yC_control - Ytrain_control, 2))

                    
                    g_treated, g_control = g[Atrain], g[1-Atrain]
                    deltaLoss = 0.5*torch.mean(torch.pow(yC_treated + g_treated - Ytrain_treated, 2))
                    deltaLoss += 0.5*torch.mean(torch.pow(yT_control - g_control - Ytrain_control, 2))
                    #deltaLoss += 1e-3*torch.mean(torch.abs(G.linear1.weight))
                    
                    #loss = 0.5*factualLoss 
                    #loss += 0.5*max(min(1, (epoch-30)/80), 0)*(deltaLoss)
#                    deltaLoss += combo['lambda']*max(min(1, (epoch-30)/80), 0)*torch.sum(torch.abs(G.linear1.weight))
#                    if sum(Ytrain_control == 0).data.cpu().numpy()[0] == 0:
#                        deltaLoss += torch.mean(torch.pow(g_control[Ytrain_control == 0], 2))
                    
                    runningLossF += factualLoss.data[0]
                    runningLossC += deltaLoss.data[0]
                    
                    loss = 0.5*factualLoss + 0.1*max(min(1, (epoch-30)/80), 0)*deltaLoss + KL/200000
                    
                    
                    
                    optG.zero_grad()
                    optEnc.zero_grad()
                    opt.zero_grad()
                    #optAE.zero_grad()
                    loss.backward()
                    
                    #optAE.step()
                    optEnc.step()
                    optG.step()
                    opt.step()
                    
                    if batchCounter % 4 == 0 and True:
                        sys.stdout.write('\rEpoch {} || Loss Factual {:.3f} || Loss Delta {:.3f} || KL {:.3f} || Adv {:.3f}'.format(epoch, 
                                              runningLossF/(batchCounter + 1), runningLossC/(batchCounter + 1),
                                              0/(batchCounter + 1), 0))
                        sys.stdout.flush()
                
            
            
            encoder.eval()
            model.eval()
            G.eval()
            h, _ = encoder(xVal)
            #h2 = torch.cat((h, xVal[:,:len(predictorBase)]), dim = 1)
            delta = G(h)
            uVal = -(delta.data.cpu().numpy())
            valAUUC = AUUC(uVal, yVal, aVal, graph = False)
            if valAUUC < auucBest:
                auucBest = valAUUC
                hyperparamsBest[j] = combo
                torch.save(encoder.state_dict(), 'models/encoder_' + str(counter) + '_' + str(j) + '.pth')
                torch.save(G.state_dict(), 'models/G_' + str(counter) + '_' + str(j) + '.pth')

            
            
    

    #xCheck, yCheck, aCheck = xTrain[predictorList[0]].values, yTrain, aTrain
    yCheck, aCheck = yTest, aTest
    
    
    uCheck = np.zeros((len(yCheck), cvIters))
    for j in range(cvIters):
        xCheck = xTest[predictorList[j]].values
        xCheck = loadVar(xCheck.astype('float32'))
        encoder = Encoder(len(predictorList[j]), 12, encodingSize).cuda()
        G = Delta(encodingSize, 4, 1).cuda()
        encoder.load_state_dict(torch.load('models/encoder_' + str(counter) + '_' + str(j) + '.pth'))   
        G.load_state_dict(torch.load('models/G_' + str(counter) + '_' + str(j) + '.pth'))
        encoder.eval()
        G.eval()
    
        h,_ = encoder(xCheck)
        #h2 = torch.cat((h, xCheck[:,:len(predictorBase)]), dim = 1)
        delta = G(h)
        uCheck[:,j] = -(delta.data.cpu().numpy())
        #h = h.data.cpu().numpy()[:,:encodingSize]
        
    allAUUCs[counter] = AUUC(np.mean(uCheck, axis = 1), yCheck, aCheck, graph = True)
    print('\n')
    #print(hyperparamsBest)
    print(allAUUCs[counter])
    print(counter)
    counter += 1