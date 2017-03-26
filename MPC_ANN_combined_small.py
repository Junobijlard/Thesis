#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:59:23 2017

@author: juno
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:25:57 2017

@author: juno
"""

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
from copy import deepcopy
from Ship import Ship
from ContainerTerminal import ContainerTerminal
import pickle
import keras

time_horizon = 6
max_time_horizon = 6
operations_cost_hour = 200
scenario = 'random'
ships_data_folder = '/Users/Juno/Desktop/Scriptie/Python/Ship configurations/test_{0}/'.format(scenario)
standardscalerpath = '/Users/Juno/Desktop/Scriptie/Python/ANN/small-{1}/standard_scaler_th{0}_{1}.p'.format(time_horizon,scenario)
ANNpathship = '/Users/Juno/Desktop/Scriptie/Python/ANN/small-{1}/ANN_ship_th{0}_{1}'.format(time_horizon,scenario)
ANNpathqc = '/Users/Juno/Desktop/Scriptie/Python/ANN/small-{1}/ANN_qc_th{0}_{1}'.format(time_horizon,scenario)
shipsFilename = 'set_of_ships_1.csv'
#==============================================================================
# Don't change!
#==============================================================================
daily_operations_cost = 24*operations_cost_hour
costList = list()
findInputTime = 0
inputTime = 0
timeSequences = 0
ANN_ship = keras.models.load_model(ANNpathship)
ANN_qc = keras.models.load_model(ANNpathqc)
dataset = pd.read_csv(ships_data_folder+shipsFilename)
standardscaler = pickle.load(open(standardscalerpath, "rb"))

def makeList(number):
    listname = list()
    for i in range(number):
        listname.append(i)
    return listname

def createJKUVXYZ(berths):
    j = [0]
    k = [0]
    u = [Ship(0, "starting ship", 0, 0)]
    v = [dict(zip(berths, [1]*len(berths)))]
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
terminal = ContainerTerminal('Groningen', 'small', 2, 7, Ship(0,'dummy', 0,0,0))
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
            y[b] = u.operating_time
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
    
def realCost():
    for ship in u_k:
        waiting_time = ship.finishing_time - ship.arrival_time
        waiting_cost = waiting_time*ship.waiting_cost
        ship.total_waiting_cost = waiting_cost
        operating_cost = ship.operating_time*daily_operations_cost
        ship.total_operating_cost = operating_cost
        
    total_operating_cost = sum([ship.total_operating_cost for ship in u_k])
    total_waiting_cost = sum([ship.total_waiting_cost for ship in u_k])
    total_cost = total_operating_cost + total_waiting_cost
    return total_cost

def findSetOfShipsToBeBerthed(time_horizon = time_horizon):
    ships_unassigned = [ship for ship in ships if ship.assigned == False]
    if len(ships_unassigned)<time_horizon:
        time_horizon = len(ships_unassigned)
    S = ships_unassigned[0:time_horizon]
    return S, time_horizon
    
def findUandV(): 
    S, time_horizon = findSetOfShipsToBeBerthed()
    inputArray = []
    currentX = deepcopy(x_k[-1])
    currentV = deepcopy(v_k[-1])
    
    V0 = [currentV[0]]
    inputArray.append(V0)
    for ship in S:
        inputArray.append(ship.training_values)
    for extra_ship in range(max_time_horizon-len(S)):
        inputArray.append([0,0,0])
    inputArray = list(chain(*inputArray))
    
    currentTime = min(currentX.values())
    j = min(currentX, key = currentX.get)
    currentY = {}
    for berth in berths:
        currentY[berth]=(currentX[berth]-currentTime)*currentV[berth]
    for berth in berths:
        inputArray.append(currentY[berth])
    
    
    X = np.array([inputArray])#.reshape(1,-1)
    X = standardscaler.transform(X)

    y_ship = ANN_ship.predict(X)
    y_qc = ANN_qc.predict(X)
    
    uIndex = y_ship[:,:time_horizon].argmax()
    u = S[uIndex]
    
    vIndex = y_qc.argmax()
    v = {}
    totalQC = 0
    for berth in berths:
        if berth!=berths[-1]:
            v[berth]=vIndex+1
            totalQC+=vIndex+1
        else:
            v[berth]=terminal.QC_number-totalQC
    return u,v    
    

def main():
    starttime = time.time()
    condition = True
    while condition == True:
        u,v = findUandV()
        u_k.append(u)
        v_k.append(v)
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
        if sum(ship.assigned == True for ship in ships)==len(ships):
            condition=False
    total_cost = realCost()
    totalTime = time.time()-starttime
    return total_cost, totalTime
    
del(ships)
th = 6
allCost = 0
allTime = 0
for i in range(20):
    totalTimeArray = []
    totalCostArray = []
    for b in range(5):
        filename = 'set_of_ships_{0}.csv'.format(i)
        ships = createShips(filename)
        j_k, k_k, u_k, v_k, x_k, y_k, z_k = createJKUVXYZ(berths)
        berthDict = {berth:[] for berth in berths}
        time1, cost1 = main()
        totalTimeArray.append(time1)
        totalCostArray.append(cost1)
    print(i)
    total_Time = np.mean(totalTimeArray)
    total_Cost = np.mean(totalCostArray)
    totalTimeArray.clear()
    totalCostArray.clear()
    allCost+=total_Cost
    allTime+=total_Time

resultDict[th]=(allCost, allTime)     

pickle.dump(resultDict,open('ANN_{0}.p'.format(scenario),"wb"))   
#resultDict = {}

#th = 6
#totalCostDictANN = {}
#totalTimeDictANN = {}
#totalCostArray = list()
#totalTimeArray = list()

#total_cost, total_time = main()
#totalCostArray.append(total_cost)
#totalTimeArray.append(total_time)

#totalCost = np.mean(totalCostArray)
#totalTime = np.mean(totalTimeArray)
#totalCostDictANN[th]=totalCost
#totalTimeDictANN[th]=totalTime
#totalCostArray.clear()
#totalTimeArray.clear()

#pickle.dump(totalCostDictANN, open('ANNcost-{0}.p'.format(scenario),"wb"))
#pickle.dump(totalTimeDictANN, open('ANNtime-{0}.p'.format(scenario),"wb"))
