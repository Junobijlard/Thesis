#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 10:03:34 2016

@author: juno
"""

from os import chdir
chdir("/Users/Juno/Desktop/Scriptie/Python")

import os
import random
import time
import pandas as pd
import math
import numpy as np
import importlib
import Initializations
import matplotlib.pyplot as plt
importlib.reload(Initializations)
from Initializations import *
from itertools import permutations
from itertools import product
from itertools import chain

"""
Parameters:
    time_horizon
    operations_cost_hour
    collect training data?
    which terminal to berth at?
    Load file and call main()
"""
time_horizon = 4
max_time_horizon = 8
operations_cost_hour = 200
terminal = terminals[-1]
collect_training_data = True
training_data_file_name = 'training_data.csv'
training_data_path = '/Users/Juno/Desktop/Scriptie/Python/Training data/'

"""
    Don't change anything from here:
"""
operations_cost = 24*operations_cost_hour
v_k = [dict(zip(berths, [0]*len(berths)))]
u_k = [Ship(0, "starting ship", 0, 0)]
costList = list()
findInputTime = 0
inputTime = 0
timeSequences = 0
training_data = []


def updateParameters(u,v):
    """
    j: earliest available berth
    x: finishing times of berths
    y: remaining operation times of berths
    z: starting times of berths
    u: input ship
    v: input QC sequence
    """
    x = {}
    y = {}
    z = {}
    lastX = x_k[-1]
    lastY = y_k[-1]
    lastZ = z_k[-1]
    lastV = v_k[-1]
    j = min(lastX, key = lastX.get)
    terminal.berths[j].finishing_time = lastX[j]
    for b in berths:
        if b==j:
            z[b]=max(lastX[b], u.arrival_time)
            y[b] = u.operation_time
            x[b] = z[b]+y[b]/v[b]
        else:
            if lastX[j]>lastZ[b]:
                z[b] = lastX[j]
            else:
                z[b] = lastZ[b]
            y[b] = lastY[b]-(z[b]-lastZ[b])*lastV[b]
            x[b] = z[b]+y[b]/v[b]
    u.assigned = True
    u.allocated_berth = j
    terminal.berths[j] = u
    berthDict[j].append(u)
    return j,x,y,z

def findSetOfShipsToBeBerthed(time_horizon = time_horizon):
    ships_unassigned = [ship for ship in ships if ship.assigned == False]
    if len(ships_unassigned)<time_horizon:
        time_horizon = len(ships_unassigned)
    S = ships_unassigned[0:time_horizon]
    return S, time_horizon

def calculatePossibleQCSequences(berths = terminal.berth_positions, QCs = terminal.QC_number):
    QClist = list()
    max_value = QCs - (berths -1)
    possible_values = list(range(1,max_value+1))
    allValuesList = [possible_values]*berths
    for possibility in product(*allValuesList):
        if sum(possibility) == QCs:
            sequence = dict(zip(list(range(berths)),possibility))
            QClist.append(sequence)
    return QClist    

def predictParameters(S, v):
    x_temp = [x_k[-1]]
    y_temp = [y_k[-1]]
    z_temp = [z_k[-1]]
    v_temp = [v_k[-1]]
    j_temp = [j_k[-1]]
    for ship in S:
        x = {}
        y = {}
        z = {}
        lastX = x_temp[-1]
        lastY = y_temp[-1]
        lastZ = z_temp[-1]
        lastV = v_temp[-1]
        lastJ = j_temp[-1]
        #voor alle boten
        j = min(lastX, key = lastX.get)
        for b in berths:
            if b == j:
                z[b] = max(lastX[b], ship.arrival_time)
                y[b] = ship.operation_time
                x[b] = z[b]+y[b]/v[b]
            else:
                if lastX[j]>lastZ[b]:
                    z[b] = lastX[j]
                else:
                    z[b] = lastZ[b]
                y[b] = lastY[b]-(z[b]-lastZ[b])*lastV[b]
                x[b] = z[b]+y[b]/v[b]
        x_temp.append(x)
        y_temp.append(y)
        z_temp.append(z)
        v_temp.append(v)
        j_temp.append(j)
    return x_temp, y_temp, z_temp, v_temp, j_temp

def costPrediction(S,v):
    x, y, z, v, j = predictParameters(S,v)
    J = 0
    for k in range(1,len(j)):
        for b in berths:
            if b == j[k]:
                J+=(x[k-1][b]-z[k-1][b])*v[k-1][b]*operations_cost
                J+=max(x[k-1][b]-S[k-1].arrival_time,0)*S[k-1].waiting_cost
            else:
                J+=(z[k][b]-z[k-1][b])*(v[k-1][b]*operations_cost+terminal.berths[b].waiting_cost)
    return J
            
def findUandV(QCList): #vinden van kosten gaat niet goed
    S, time_horizon = findSetOfShipsToBeBerthed()
    S_combinations = [possibility for possibility in permutations(S, time_horizon)]
    costList = list()
    for S_possibility in S_combinations:
        for QC_possibility in QCList:
            costList.append((S_possibility, QC_possibility, costPrediction(S_possibility, QC_possibility)))
    minimum_cost = min(costList, key = lambda t: t[2])
    u = minimum_cost[0][0]
    v = minimum_cost[1]
    if collect_training_data == True:
        training_data_to_append = {}
        training_data_to_append['x'] = x_k[-1]
        training_data_to_append['S'] = S
        training_data_to_append['u'] = u
        training_data_to_append['v'] = v
        global training_data
        training_data.append(training_data_to_append)
    return u,v    
    
def realCost(u):
    J = 0
    for b in berths:
        if b == j_k[-1]:
            J+=(x_k[-1][b]-z_k[-1][b])*v_k[-1][b]*operations_cost
            J+=max(x_k[-1][b]-u.arrival_time,0)*u.waiting_cost
        else:
            J+=(z_k[-1][b]-z_k[-2][b])*(v_k[-2][b]*operations_cost+terminal.berths[b].waiting_cost)
    return J
    
def writeTrainingDataCSV(filename = training_data_file_name, path = training_data_path, training_data = training_data, max_time_horizon = max_time_horizon):
    columns = []
    for berth in berths:
        columns.append('X {0}'.format(berth))
    
    for i in range(max_time_horizon):
        name1 = 'Ship {0} arrival time'.format(i+1)
        name2 = 'Ship {0} teu'.format(i+1)
        name3 = 'Ship {0} waiting cost'.format(i+1)
        columns.append(name1)
        columns.append(name2)
        columns.append(name3)
    columns.append('U')
    
    for berth in berths:
        columns.append('V {0}'.format(berth))
    
    all_data = []
    for line in range(len(training_data)):
        x = training_data[line]['x']
        S = training_data[line]['S']
        u = training_data[line]['u']
        v = training_data[line]['v']
        
        x_data = list(x.values())
        S_data = [ship.training_values for ship in S]
        for i in range(max_time_horizon-len(S)):
            S_data.append([0,0,0])
        S_data = list(chain(*S_data))
        u_data = [S.index(u)]
        v_data = list(v.values())
        total_data = [x_data, S_data, u_data, v_data]
        total_data = list(chain(*total_data))
        total_data_dictionary = dict(zip(columns, total_data))
        all_data.append(total_data_dictionary)
    training_data_dataframe = pd.DataFrame(all_data)
    pathOfTrainingFile = '{0}{1}'.format(path, filename)
    if os.path.isfile(pathOfTrainingFile):
        training_data_dataframe.to_csv(pathOfTrainingFile, index = False, mode = 'a', header = False)
    else:
        training_data_dataframe.to_csv(pathOfTrainingFile, index = False)
    
def main():
    starttime = time.time()
    totalCost = 0
    inputTime = 0
    updateTime = 0
    sequencesStartTime = time.time()
    QCList = calculatePossibleQCSequences()
    sequencesTime = time.time()-sequencesStartTime
    condition = True
    while condition == True:
        inputStartTime = time.time()
        u,v = findUandV(QCList)
        u_k.append(u)
        v_k.append(v)
        inputTime += (time.time()-inputStartTime)
        parameterStartTime = time.time()
        j,x,y,z = updateParameters(u,v)
        global x_k
        x_k.append(x)
        global y_k
        y_k.append(y)
        global z_k
        z_k.append(z)
        global j_k
        j_k.append(j)
        updateTime += (time.time()-parameterStartTime)
        totalCost += realCost(u)
        if sum(ship.assigned == True for ship in ships)==len(ships):
            condition=False
    totalTime = time.time()-starttime
    print("Total cost: \t", totalCost)
    print("Calculation time: ", totalTime)
    print("\t Time to calculate QC sequences: ", sequencesTime)
    print("\t Time to calculate inputs: \t", inputTime)
    print("\t Time to update parameters: \t", updateTime)
    times = {"totalTime": totalTime, "sequencesTime": sequencesTime, "inputTime": inputTime, "updateTime": updateTime }
    if collect_training_data == True:
        writeTrainingDataCSV()
    return times
    
    

    