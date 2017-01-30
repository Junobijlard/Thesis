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

from os import chdir
chdir('/Users/juno/Desktop/Scriptie/Python')
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

#==============================================================================
# PARAMETERS
#==============================================================================
training_data = 'training_data-NEW.csv'
training_data_path = '/Users/juno/Desktop/Scriptie/Python/Training data/'
number_of_ships = 5
num_hidden_layers = 2
num_nodes = 20
test_size = 0.2
#importing the dataset

def preprocessData1TimeHorizon(test_size = test_size):
    dataset = pd.read_csv(training_data_path+training_data)
    dataset = dataset[dataset['Ship 5 arrival time']!=0]
    allVandU = [i for i in dataset.columns if 'V' in i or 'U' in i]
    allShipsandX = [i for i in dataset.columns if 'Ship' in i or 'X' in i]
    X = dataset.drop(allVandU, axis = 1).values
    y = dataset.drop(allShipsandX, axis = 1)
    y = y.drop('V 1', axis = 1).values #deze regel verwijderen en .values bij regel hierboven toevoegen
    onehotencoder = OneHotEncoder()
    y = onehotencoder.fit_transform(y).toarray() #Transforms y to binary array
    standardscaler = StandardScaler()
    X = standardscaler.fit_transform(X) #Scales values of X 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size, random_state = 0)
    return X_train, X_test, y_train, y_test, standardscaler
    
def preprocessData(test_size = test_size):
    dataset = pd.read_csv(training_data_path+training_data)
    dataset = dataset.sample(n=10000)
    allVandU = [i for i in dataset.columns if 'V' in i and 'Current' not in i or 'U' in i]
    allShipsandX = [i for i in dataset.columns if 'Ship' in i or 'X' in i or 'Current' in i]
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
    network.compile(optimizer = 'Nadam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    network.fit(X_train,
            y_train,
            batch_size = 10,
            nb_epoch = 100)
    return network

def findAccuracy(y_pred, y_test, number_of_ships = number_of_ships):
    ship_pred = y_pred[:,:number_of_ships]
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
    X_train, X_test, y_train, y_test, standardscaler = preprocessData()
    starttime = time.time()
    ANN = buildANN(X_train, X_test, y_train, y_test)
    y_pred = ANN.predict(X_test)
    ship_accuracy, QC_accuracy = findAccuracy(y_pred, y_test)
    print(time.time()-starttime)
    return ANN, standardscaler,ship_accuracy, QC_accuracy
