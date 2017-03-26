#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:32:33 2017

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

time_horizon = 1
max_time_horizon = 6
operations_cost_hour = 200
collect_training_data = False
scenario = 'equal'

training_data_path = '/Users/Juno/Desktop/Scriptie/Python/Test data/{0}-th{1}.csv'.format(scenario,time_horizon)
ships_data_folder = '/Users/Juno/Desktop/Scriptie/Python/Ship configurations/test_{0}/'.format(scenario)
shipsFilename = 'set_of_ships_1.csv'
LB_training_data = 0
UB_training_data = LB_training_data+20

daily_operations_cost = 24*operations_cost_hour
costList = list()
findInputTime = 0
inputTime = 0
timeSequences = 0
training_data = []


dataset = pd.read_csv(ships_data_folder+shipsFilename)

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
             

def predictCosts(S, v, currentY, currentTime, j):
    lastX_copy = dict(zip(berths, [currentTime]*len(berths)))
    total_cost = 0
    for berth in berths:
        if berth!=j:
            lastX_copy[berth]=currentTime+currentY[berth]/v[berth]
            ship = terminal.berths[berth]
            waiting_cost = (lastX_copy[berth]-ship.arrival_time)*ship.waiting_cost
            total_cost+=waiting_cost
    for ship in S:
        j = min(lastX_copy, key = lastX_copy.get)
        starting_time = max(lastX_copy[j], ship.arrival_time)
        finishing_time = starting_time + ship.operating_time/v[j]
        lastX_copy[j] = finishing_time
        waiting_cost = (finishing_time - ship.arrival_time) * ship.waiting_cost
        total_cost+= waiting_cost
    return total_cost   
  
def findInputs(QCList):
    S,time_horizon  = findSetOfShipsToBeBerthed()
    currentX = deepcopy(x_k[-1])
    currentV = deepcopy(v_k[-1])
    currentTime = min(currentX.values())
    j = min(currentX, key = currentX.get)
    currentY = {}
    for berth in berths:
        currentY[berth]=(currentX[berth]-currentTime)*currentV[berth]    
    #calculate cost per possibility
    S_combinations = [possibility for possibility in permutations(S, time_horizon)]
    costList = list()
    for S_possibility in S_combinations:
        for QC_possibility in QCList:
            total_cost = predictCosts(S_possibility, QC_possibility, currentY, currentTime, j)
            toAppend = (S_possibility, QC_possibility, total_cost)
            costList.append(toAppend)
    minimum_cost = min(costList, key = lambda t: t[2])
    u = minimum_cost[0][0]
    v = minimum_cost[1]
    if collect_training_data == True:
        training_data_to_append = {}
        training_data_to_append['y'] = list(currentY.values())
        training_data_to_append['S'] = S
        training_data_to_append['current v']=v_k[-1]
        training_data_to_append['u'] = u
        training_data_to_append['v'] = v
        global training_data
        training_data.append(training_data_to_append)
    return u,v 
    
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

def writeTrainingDataCSV(training_data_path = training_data_path, training_data = training_data, max_time_horizon = max_time_horizon):
    columns = []
    for berth in berths:
        columns.append('y {0}'.format(berth))
    
    for i in range(max_time_horizon):
        name1 = 'Ship {0} arrival time'.format(i+1)
        name2 = 'Ship {0} teu'.format(i+1)
        name3 = 'Ship {0} waiting cost'.format(i+1)
        columns.append(name1)
        columns.append(name2)
        columns.append(name3)
        
    for berth in berths:
        columns.append('Current V {0}'.format(berth))
    columns.append('U')
    
    for berth in berths:
        columns.append('V {0}'.format(berth))
    
    all_data = []
    for line in range(len(training_data)):
        y = training_data[line]['y']
        S = training_data[line]['S']
        currentV = training_data[line]['current v']
        u = training_data[line]['u']
        v = training_data[line]['v']
        
        y_data = y
        S_data = [ship.training_values for ship in S]
        for i in range(max_time_horizon-len(S)):
            S_data.append([0,0,0])
        S_data = list(chain(*S_data))
        currentV_data = list(currentV.values())
        u_data = [S.index(u)]
        v_data = list(v.values())
        total_data = [y_data, S_data, currentV_data, u_data, v_data]
        total_data = list(chain(*total_data))
        total_data_dictionary = dict(zip(columns, total_data))
        all_data.append(total_data_dictionary)
        training_data_dataframe = pd.DataFrame(all_data)
    if os.path.isfile(training_data_path):
        training_data_dataframe.to_csv(training_data_path, index = False, mode = 'a', header = False)
    else:
        training_data_dataframe.to_csv(training_data_path, index = False)
    
def main():
    starttime = time.time()
    inputTime = 0
    updateTime = 0
    sequencesStartTime = time.time()
    QCList = calculatePossibleQCSequences()
    sequencesTime = time.time()-sequencesStartTime
    condition = True
    global training_data
    training_data.clear()
    while condition == True:
        inputStartTime = time.time()
        u,v = findInputs(QCList)
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
    if collect_training_data == True:
        writeTrainingDataCSV()
    return totalTime, total_cost
    

if collect_training_data:
    for i in range(LB_training_data,UB_training_data):
        filename = 'set_of_ships_{0}.csv'.format(i)
        ships = createShips(filename)
        j_k, k_k, u_k, v_k, x_k, y_k, z_k = createJKUVXYZ(berths)
        berthDict = {berth:[] for berth in berths}
        main()
        print(i)

"""del(ships)
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

pickle.dump(resultDict,open('MPC_{0}.p'.format(scenario),"wb"))   
#resultDict = {}"""

def graph(timeDict, title, scale, saveName, color):
    n_groups = max(timeDict)
    fig = plt.figure()
    values = list(timeDict.values())
    color = color
    opacity = 0.4
    index = np.arange(n_groups)+0.5
    bar_width = 0.35
    bars = plt.bar(index, values, bar_width,
                 alpha=opacity,
                 color=color)
    plt.xlabel('Time horizon')
    plt.yscale(scale)
    plt.ylabel('Calculation time (s)')
    plt.xticks(index + bar_width / 2, (1,2,3,4,5,6))
    #plt.tight_layout
    plt.grid()
    plt.title(title)
    if scale == 'linear':
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if saveName:
        plt.savefig(saveName, bbox_inches='tight')
    plt.show()
    

        