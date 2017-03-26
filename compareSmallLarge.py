#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:58:19 2017

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

scenario = 'equal'
number_of_ships = 6
time_horizon=1

def calculateAccuracies(scenario, time_horizon):
    largeANNpath = '/Users/juno/Desktop/Scriptie/Python/ANN/{0}'.format(scenario)
    largeANN = keras.models.load_model(largeANNpath)
    smallANNpathship = '/Users/juno/Desktop/Scriptie/Python/ANN/small-{1}/ANN_ship_th{0}_{1}'.format(time_horizon,scenario)
    ANN_ship = keras.models.load_model(smallANNpathship)
    smallANNpathqc = '/Users/juno/Desktop/Scriptie/Python/ANN/small-{1}/ANN_qc_th{0}_{1}'.format(time_horizon,scenario)
    ANN_qc = keras.models.load_model(smallANNpathqc)
    
    largestandardscalerpath = '/Users/juno/Desktop/Scriptie/Python/ANN/standard_scaler_{0}.p'.format(scenario)
    smallstandardscalerpath = '/Users/juno/Desktop/Scriptie/Python/ANN/small-{1}/standard_scaler_th{0}_{1}.p'.format(time_horizon, scenario)
    small_standard_scaler = pickle.load(open(smallstandardscalerpath, "rb"))
    large_standard_scaler = pickle.load(open(largestandardscalerpath, "rb"))
    
    dataset = pd.read_csv('/Users/juno/Desktop/Scriptie/Python/Test data/{0}-th{1}.csv'.format(scenario,time_horizon))
    if time_horizon:
        dataset = dataset.loc[dataset['Ship {0} arrival time'.format(time_horizon)]!=0]
    if time_horizon!= 6:
        dataset = dataset.loc[dataset['Ship {0} arrival time'.format(time_horizon+1)]==0]
    dataset = dataset.loc[:49,:]                                
    allVandU = [i for i in dataset.columns if 'V' in i and 'Current' not in i or 'U' in i]
    allShipsandX = [i for i in dataset.columns if 'Ship' in i or 'y' in i or 'Current' in i]
    
    X = dataset.drop(allVandU, axis = 1)
    X = X.drop('Current V 1', axis = 1)
    X = X.values
    Xsmall = small_standard_scaler.transform(X)
    Xlarge = large_standard_scaler.transform(X)
    
    y = dataset.drop(allShipsandX, axis = 1)
    y = y.drop('V 1', axis = 1).values
    y_ship = y[:,0]
    y_qc = y[:,1]
    
    y_pred_ship_small = ANN_ship.predict(Xsmall)
    y_pred_ship_small = [i.argmax() for i in y_pred_ship_small]
    ship_accuracy_small = round(accuracy_score(y_ship, y_pred_ship_small)*100,1)
    
    y_pred_qc_small = ANN_qc.predict(Xsmall)
    y_pred_qc_small = [i.argmax()+1 for i in y_pred_qc_small]
    qc_accuracy_small = round(accuracy_score(y_qc, y_pred_qc_small)*100,1)
    
    y_large = largeANN.predict(Xlarge)
    large_ship_pred = y_large[:,:number_of_ships]
    if time_horizon:
        large_ship_pred = large_ship_pred[:,:time_horizon]
    large_ship_pred = [i.argmax() for i in large_ship_pred]
    
    ship_accuracy_large = round(accuracy_score(y_ship, large_ship_pred)*100,1)
    
    QC_pred_large = y_large[:,number_of_ships:]
    QC_pred_large = [i.argmax()+1 for i in QC_pred_large]
    
    qc_accuracy_large = round(accuracy_score(y_qc, QC_pred_large)*100,1)
    return ship_accuracy_small, qc_accuracy_small, ship_accuracy_large, qc_accuracy_large

smallDict = {}
largeDict = {}
for i in range(1,7):
    time_horizon = i
    ship_accuracy_small, qc_accuracy_small, ship_accuracy_large, qc_accuracy_large = calculateAccuracies(scenario, time_horizon)
    smallTuple = (ship_accuracy_small, qc_accuracy_small)
    largeTuple = (ship_accuracy_large, qc_accuracy_large)
    smallDict[i]=smallTuple
    largeDict[i] = largeTuple
pickle.dump(smallDict, open("smallDict.p", "wb"))
pickle.dump(largeDict, open("largeDict.p","wb"))
    