#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:19:40 2017

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
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import product
from itertools import chain
from predictCostClass import predictCostClass
from copy import deepcopy
from Ship import Ship
from ContainerTerminal import ContainerTerminal

time_horizon = 4
max_time_horizon = 8
operations_cost_hour = 200
collect_training_data = False
training_data_file_name = 'training_data.csv'
training_data_path = '/Users/Juno/Desktop/Scriptie/Python/Training data/'
ships_data_folder = '/Users/Juno/Desktop/Scriptie/Python/Ship configurations/'
shipsFilename = 'ships_bayu.csv'
#==============================================================================
# Don't change!
#==============================================================================
operations_cost = 24*operations_cost_hour
costList = list()
findInputTime = 0
inputTime = 0
timeSequences = 0
training_data = []

def makeList(number):
    listname = list()
    for i in range(number):
        listname.append(i)
    return listname

def createJKUVXYZ(berths):
    j = [0]
    k = [0]
    u = [Ship(0, "starting ship", 0, 0)]
    v = [dict(zip(berths, [0]*len(berths)))]
    x = [dict(zip(berths, [0]*len(berths)))]
    y = [dict(zip(berths, [0]*len(berths)))]
    z = [dict(zip(berths, [0]*len(berths)))]
    return j,k,u,v,x,y,z
    
def createShips(filename):
    total_path = ships_data_folder+filename
    shipsDF = pd.read_csv(total_path)
    ships = list()
    for i in range(len(shipsDF)):
        name = shipsDF['Name'][i]
        arrival_time = shipsDF['Arrival Time'][i]
        teu = shipsDF['TEU'][i]
        waiting_cost = shipsDF['Waiting Cost'][i]
        ship = Ship(i, name, arrival_time, teu, waiting_cost)
        ships.append(ship)
    return ships

def createTerminals(filename = 0):
    terminals = list()
    ship = Ship(0,'dummy', 0,0,0)
    if filename == 0:
        terminal = ContainerTerminal('Groningen', 'small', 2, 7, ship)
    terminals.append(terminal)
    return terminals
    
ships = createShips(shipsFilename)
terminals = createTerminals()
terminal = terminals[-1]
berths = makeList(terminal.berth_positions)
j_k, k_k, u_k, v_k, x_k, y_k, z_k = createJKUVXYZ(berths)
berthDict = {berth:[] for berth in berths}
             
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
    k = min(lastX.values())
    u.starting_time = lastX[j]
    #terminal.berths[j].finishing_time = lastX[j]
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
    for berth in berths:
        terminal.berths[berth].finishing_time = x[berth]
    berthDict[j].append(u)
    return k,j,x,y,z

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

def predictCost(S,v):
    lastX_copy = deepcopy(x_k[-1])
    lastY_copy = deepcopy(y_k[-1])
    costDict = {}
    costClass = 0
    current_time = min(lastX_copy.values())
    j = min(lastX_copy, key = lastX_copy.get)
    for berth in berths:
        if berth != j:
            remaining_operating_time = (lastX_copy[berth]-current_time)*v_k[-1][berth]/v[berth]
            lastX_copy[berth] = remaining_operating_time+current_time
    for counter, ship in enumerate(S):
        j = min(lastX_copy, key = lastX_copy.get)
        starting_time = max(lastX_copy[j], ship.arrival_time)
        costClass = predictCostClass(ship, v[j], starting_time, operations_cost)
        costDict[counter]= costClass.total_cost
        lastX_copy[j] = costClass.finishing_time
    total_cost = sum(costDict.values())
    return total_cost
        
            
def findUandV(QCList): 
    S, time_horizon = findSetOfShipsToBeBerthed()
    S_combinations = [possibility for possibility in permutations(S, time_horizon)]
    costList = list()
    for S_possibility in S_combinations:
        for QC_possibility in QCList:
            costList.append((S_possibility, QC_possibility, predictCost(S_possibility, QC_possibility)))
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
    
def realCost():
    for ship in u_k:
        for i in range(len(k_k)):
            if k_k[i] > ship.starting_time and k_k[i]<=ship.finishing_time:
                time = k_k[i]-k_k[i-1]
                berth = ship.allocated_berth
                ship.cost_for_operation+=time*v_k[i][berth]*operations_cost
        ship.cost_for_waiting = (ship.finishing_time - ship.arrival_time)*ship.waiting_cost
        
    total_operating_cost = sum([ship.cost_for_operation for ship in u_k])
    total_waiting_cost = sum([ship.cost_for_waiting for ship in u_k])
    total_cost = total_operating_cost + total_waiting_cost
    return total_cost

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
        k,j,x,y,z = updateParameters(u,v)
        global x_k
        x_k.append(x)
        global y_k
        y_k.append(y)
        global z_k
        z_k.append(z)
        global j_k
        j_k.append(j)
        global k_k
        k_k.append(k)
        updateTime += (time.time()-parameterStartTime)
        if sum(ship.assigned == True for ship in ships)==len(ships):
            condition=False
    total_cost = realCost()
    totalTime = time.time()-starttime
    print("Total cost: \t", totalCost)
    print("Calculation time: ", totalTime)
    print("\t Time to calculate QC sequences: ", sequencesTime)
    print("\t Time to calculate inputs: \t", inputTime)
    print("\t Time to update parameters: \t", updateTime)
    times = {"totalTime": totalTime, "sequencesTime": sequencesTime, "inputTime": inputTime, "updateTime": updateTime }
    if collect_training_data == True:
        writeTrainingDataCSV()
    return total_cost