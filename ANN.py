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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from Ship import Ship
import keras
from keras.models import Sequential 
from keras.layers import Dense 
from sklearn.metrics import accuracy_score
#==============================================================================
# PARAMETERS
#==============================================================================
training_data = 'Training_data.csv'
training_data_path = '/Users/juno/Desktop/Scriptie/Python/Training data/'
number_of_ships = 5
number_of_QC = 7
#importing the dataset
dataset = pd.read_csv(training_data_path+training_data)
X = dataset.drop(['U', 'V 0', 'V 1'], axis = 1).values
y = dataset.iloc[:,15:17].values #dropped value of 2nd berth

# OneHotEncode y
onehotencoderY = OneHotEncoder(categorical_features=[0,1])
y = onehotencoderY.fit_transform(y).toarray()

# dummy variable trap?
#y = y[:,1:]
#traintestsplit: from sklearn.model_selection import train_test_split
num_nodes = int(np.mean([X.shape[-1], y.shape[-1]]))
num_layers = 3
#feature scaling
sc = StandardScaler()
X = sc.fit_transform(X)

#split data into test and training set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#Building the ANN
#Initializing the model

def buildANN(X_train, X_test, y_train, y_test, num_layers = 3):
    network = Sequential()
    input_layer = Dense(output_dim = num_nodes, init = 'uniform', activation = 'relu', input_dim = X.shape[-1])
    network.add(input_layer)
    for i in range(num_layers-2):
        layer = Dense(output_dim = num_nodes, init = 'uniform', activation = 'relu')
        network.add(layer)
    output_layer = Dense(output_dim = y.shape[-1], init = 'uniform', activation = 'softmax')
    network.add(output_layer)
    network.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    network.fit(X_train,
            y_train,
            batch_size = 10,
            nb_epoch = 100,
            validation_data = (X_test, y_test))
    return network


y_pred = classifier.predict(X_test)

def findAccuracy(y_pred, number_of_ships = number_of_ships):
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
    
