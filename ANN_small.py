#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:11:20 2017

@author: juno
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:22:20 2017

@author: juno
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from Ship import Ship
import keras
from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from Ship import Ship
from ContainerTerminal import ContainerTerminal
import pickle

#==============================================================================
# PARAMETERS
#==============================================================================

number_of_ships = 6
num_hidden_layers = 2
num_nodes = 100
scenario = 'equal'
training_data = '/Users/juno/Desktop/Scriptie/Python/Training data/Groningen-{0}.csv'.format(scenario)
save = True
test_size = 0.2
QCs = 7
sample = 10000
time_horizon = 6
optimizer = 'SGD'
#ANNpath = '/Users/Juno/Desktop/Scriptie/Python/ANN/th1-{0}'.format(scenario)
#ANN = keras.models.load_model(ANNpath)

#importing the dataset
def makeList(number):
    listname = list()
    for i in range(number):
        listname.append(i)
    return listname
    
def fitOneHotEncoder(QCs=QCs):
    QCListOHE = [[QC]for QC in range(1,QCs)]
    onehotencoder = OneHotEncoder()
    onehotencoder.fit_transform(QCListOHE)    
    return onehotencoder

def preprocessData(test_size = test_size, sample = sample):
    dataset = pd.read_csv(training_data)
    if time_horizon:
        dataset = dataset.loc[dataset['Ship {0} arrival time'.format(time_horizon)]!=0]
    if time_horizon!= 6:
        dataset = dataset.loc[dataset['Ship {0} arrival time'.format(time_horizon+1)]==0]
    if sample:
        dataset = dataset.sample(n = sample)
    allVandU = [i for i in dataset.columns if 'V' in i and 'Current' not in i or 'U' in i]
    allShipsandX = [i for i in dataset.columns if 'Ship' in i or 'y' in i or 'Current' in i]
    X = dataset.drop(allVandU, axis = 1)
    X = X.drop('Current V 1', axis = 1)
    X = X.values
    y = dataset.drop(allShipsandX, axis = 1)
    y = y.drop('V 1', axis = 1).values #deze regel verwijderen en .values bij regel hierboven toevoegen
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size, random_state = 0)
    y_train_qc = y_train[:,1]
    y_train_ship = y_train[:,0]
    y_test_qc =y_test[:,1]
    y_test_ship = y_test[:,0]
    
    onehotencoder_ship = OneHotEncoder()
    shipArray = np.arange(number_of_ships).reshape(-1,1)
    shipArray = onehotencoder_ship.fit_transform(shipArray).toarray()
    
    y_train_ship = y_train_ship.reshape(-1,1)
    y_train_ship = onehotencoder_ship.transform(y_train_ship).toarray()
    
    y_test_ship = y_test_ship.reshape(-1,1)
    y_test_ship = onehotencoder_ship.transform(y_test_ship).toarray()
    
    
    onehotencoder_qc = OneHotEncoder()
    qcArray = np.arange(1,QCs).reshape(-1,1)
    qcArray = onehotencoder_qc.fit_transform(qcArray).toarray()
    
    y_train_qc = y_train_qc.reshape(-1,1)
    y_train_qc = onehotencoder_qc.transform(y_train_qc).toarray()
    y_test_qc = y_test_qc.reshape(-1,1)
    y_test_qc = onehotencoder_qc.transform(y_test_qc).toarray()
    
    standardscaler = StandardScaler()
    X_train = standardscaler.fit_transform(X_train) #Scales values of X 
    X_test = standardscaler.transform(X_test)
    return X_train, X_test, y_train_ship, y_test_ship, y_train_qc, y_test_qc, standardscaler

def buildANN(X_train, X_test, y_train, y_test,num_hidden_layers =num_hidden_layers, num_nodes = num_nodes):
    network = Sequential()
    input_layer = Dense(output_dim = num_nodes, init = 'uniform', activation = 'relu', input_dim = X_train.shape[-1])
    network.add(input_layer)
    for i in range(num_hidden_layers):
        layer = Dense(output_dim = num_nodes, init = 'uniform', activation = 'relu')
        network.add(layer)
    output_layer = Dense(output_dim = y_train.shape[-1], init = 'uniform', activation = 'softmax')
    network.add(output_layer)
    network.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    network.fit(X_train,
            y_train,
            batch_size = 10,
            nb_epoch = 100)
    return network

def findAccuracy(y_pred, y_test, number_of_ships = number_of_ships):
    y_pred = [ship.argmax() for ship in y_pred]
    y_test = [ship.argmax() for ship in y_test]
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
    
def main():
    X_train, X_test, y_train_ship, y_test_ship,y_train_qc, y_test_qc, standardscaler = preprocessData()
    starttime = time.time()
    ANN_ship = buildANN(X_train, X_test, y_train_ship, y_test_ship)
    ANN_qc = buildANN(X_train, X_test, y_train_qc, y_test_qc)
    y_pred_ship = ANN_ship.predict(X_test)
    y_pred_qc = ANN_qc.predict(X_test)
    ship_accuracy = findAccuracy(y_pred_ship, y_test_ship)
    QC_accuracy = findAccuracy(y_pred_qc, y_test_qc)
    print(time.time()-starttime)
    return ANN_ship, ANN_qc, standardscaler,ship_accuracy, QC_accuracy

ANN_ship, ANN_qc, standardscaler, ship_accuracy, QC_accuracy = main()
print('Ship accuracy: ', ship_accuracy)
print('QC accuracy: ', QC_accuracy)
print(round(ship_accuracy*100,1), '\% & ',round(QC_accuracy*100,1),'\%  &')

if save:
    ANN_ship.save('ANN/small-{1}/ANN_ship_th{0}_{1}'.format(time_horizon,scenario))
    ANN_qc.save('ANN/small-{1}/ANN_qc_th{0}_{1}'.format(time_horizon,scenario))
    pickle.dump(standardscaler, open("ANN/small-{1}/standard_scaler_th{0}_{1}.p".format(time_horizon,scenario), "wb"))
    
