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


#==============================================================================
# Parameters
#==============================================================================
"""
training_data = 'Training_data.csv'
training_data_path = '/Users/juno/Desktop/Scriptie/Python/Training data/'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelEncoder
from Ship import Ship

os.chdir('/Users/juno/Desktop/Scriptie/Python')

#importing the dataset
dataset = pd.read_csv(training_data_path+training_data)
#X = 
#y = 

# OneHotEncode y
# delete 1 row to avoid dummy variable trap: y = y[:,1:]

#traintestsplit: from sklearn.model_selection import train_test_split

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

#Building the ANN
import keras
from keras.models import Sequential #initialize model
from keras.layers import Dense #add layers

#Initializing the model
classifier = Sequential()

#Adding the input layer and the first hidden layer
#output_dim = number of nodes in output layer. rule of thumb: average of number of nodes in input layer and number of nodes in output layer
# init: random weights for initializing
# relu = rect activation function? 
# input_dim = number of input variables in input layer
layer1 = Dense(output_dim = 17, init= 'uniform', activation = 'relu', input_dim = 26) 
classifier.add(layer1)
layer2 = Dense(output_dim = 17, init = 'uniform', activation = 'relu')
classifier.add(layer2)
outputlayer = Dense(output_dim = 3, init = 'uniform', activation = 'softmax')
classifier.add(outputlayer)


classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the model to the dataset
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)



import pandas as pd
dataframe = pd.read_csv('/Users/juno/Desktop/Scriptie/Python/Training data/training_data.csv')
