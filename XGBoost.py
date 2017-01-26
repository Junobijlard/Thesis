#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:16:50 2017

@author: juno
"""

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
from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score
#==============================================================================
# PARAMETERS
#==============================================================================
training_data = 'Training_data(small).csv'
training_data_path = '/Users/juno/Desktop/Scriptie/Python/Training data/'
number_of_ships = 5
num_layers = 3
num_nodes = 20
test_size = 0.2

def preprocessData(test_size = test_size):
    dataset = pd.read_csv(training_data_path+training_data)
    allVandU = [i for i in dataset.columns if 'V' in i or 'U' in i]
    allShipsandX = [i for i in dataset.columns if 'Ship' in i or 'X' in i]
    X = dataset.drop(allVandU, axis = 1).values
    y = dataset.drop(allShipsandX, axis = 1)
    y = y.drop('V 1', axis = 1).values #deze regel verwijderen en .values bij regel hierboven toevoegen
    onehotencoder = OneHotEncoder()
    y = onehotencoder.fit_transform(y).toarray() #Transforms y to binary array
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size, random_state = 0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocessData()

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(x_test)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.stdev()
