#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:22:20 2017

@author: juno
"""

# data preprocessing

# from sklearn.preprocessing import OneHotEncoder #gebruik om 1,2,3,4 in [1000, 0100, 0010, 0001] om te zetten
# onehotencoder = OneHotEncoder(categorical_featurs = [0]) #[0] = column to encode
# X = onehotencoder.fit_transform(X).toarray()
# Dummy variable trap: linear regression part. Always omit 1 dummy variable
# Remove 1 of the dummy variables
# Feature scaling om alle gegevens in dezelfde range te krijgen -> door gebruik Euclidean Distance . Niet per se voor 
# Apart voor x en y?
# from sklearn.preprocess import StandardScaler
# sc = StandardScaler()

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
num_hidden_layers = 1
num_nodes = 50
scenario = 'equal'
training_data = '/Users/juno/Desktop/Scriptie/Python/Training data/Groningen-{0}.csv'.format(scenario)
save = True
test_size = 0.2
QCs = 7
sample = False
time_horizon = 1
optimizer = 'SGD'
ANNpath = '/Users/Juno/Desktop/Scriptie/Python/ANN/th1-{0}'.format(scenario)
ANN = keras.models.load_model(ANNpath)

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
    
def preprocessData(test_size = test_size, sample = False):
    dataset = pd.read_csv(training_data)
    if time_horizon:
        index = 3*time_horizon
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
    onehotencoder = OneHotEncoder()
    y = onehotencoder.fit_transform(y).toarray() #Transforms y to binary array
    standardscaler = StandardScaler()
    X = standardscaler.fit_transform(X) #Scales values of X 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size, random_state = 0)
    return X_train, X_test, y_train, y_test, standardscaler


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
    ship_pred = y_pred[:,:number_of_ships]
    if time_horizon:
        ship_pred = ship_pred[:,:time_horizon]
    ship_pred = [i.argmax() for i in ship_pred]
    ship_test = y_test[:,:number_of_ships]
    ship_test = [i.argmax() for i in ship_test]
    ship_accuracy = accuracy_score(ship_test, ship_pred)
    
    QC_pred = y_pred[:,number_of_ships:]
    QC_pred = [i.argmax() for i in QC_pred]
    QC_test = y_test[:, number_of_ships:]
    QC_test = [i.argmax() for i in QC_test]
    QC_accuracy = accuracy_score(QC_test, QC_pred)
    
    return ship_accuracy, QC_accuracy
    
def main():
    X_train, X_test, y_train, y_test, standardscaler = preprocessData(sample = sample)
    starttime = time.time()
    ANN = buildANN(X_train, X_test, y_train, y_test)
    y_pred = ANN.predict(X_test)
    ship_accuracy, QC_accuracy = findAccuracy(y_pred, y_test)
    print(time.time()-starttime)
    return ANN, standardscaler,ship_accuracy, QC_accuracy

ANN, standardscaler, ship_accuracy, QC_accuracy = main()
print('Ship accuracy: ', ship_accuracy)
print('QC accuracy: ', QC_accuracy)

if save:
    ANN.save('th1-{0}'.format(scenario))
    pickle.dump(standardscaler, open("standard_scaler_th1_{0}.p".format(scenario), "wb"))
    
