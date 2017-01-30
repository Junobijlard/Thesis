#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 22:01:39 2016

@author: Juno
"""
import numpy as np
import pandas as pd
import random
import math
#==============================================================================
# PARAMETERS
# load_type: load_scenario
# number_of_ships: number of ships to create
# path of save file
# name of save file
# m = mean of ship interarrival time
# v = standard deviation of ship interarrival time
# TEU_UB: upper bound of number of containers on ship in scenario
# TEU_LB: lower bound of number of containers on ship in scenario
#==============================================================================
load_type = 1 #reduced = 1, normal = 2, heavy = 3
number_of_ships = 50
nameOfFile = 'test.csv'

path = '/Users/Juno/Desktop/Scriptie/Python/Ship configurations/Large Horizon'
pathOfNames = '/Users/Juno/Desktop/Scriptie/Python/Ship configurations/Ship_names.csv'
waiting_cost_mean = 3600
waiting_cost_stdev = 300

#==============================================================================
# CODE
#==============================================================================
Waiting_LB = 3000
Waiting_UB = 4000

if load_type == 1:
    m = 6
    v = 0.6
    TEU_UB=3000
    TEU_LB=1000
elif load_type == 2:
    m = 5.5
    v = 0.5
    TEU_UB = 5000
    TEU_LB = 3000
else:
    m = 5.0
    v = 0.4
    TEU_UB = 10000
    TEU_LB = 5000

m = m/24
v = v/24

names = pd.read_csv(pathOfNames)
pathOfShipFile = '{0}{1}'.format(path, nameOfFile)

class Ship(object):
    """ Class used to generate a ship with attributes:
    name: name of the ship
    load type: type of load
    arrival_time: arrival time of the ship in days
    TEU: number of TEUs the ship is transporting
    operation time: the number of QC minutes it takes to unload the ship"""
    
    def __init__(self, ship_number, name, arrival_time, TEU, waiting_cost = 3600):
        self.ship_number = ship_number
        self.name = name
        self.arrival_time = arrival_time
        self.TEU = TEU
        operation = TEU*3/60/24
        self.operating_time = round(operation,2)
        self.waiting_cost = waiting_cost
        self.allocated_berth = "-1"
        self.assigned = False
        self.starting_time = 0
        self.finishing_time = 0
        self.training_values = [self.arrival_time, self.TEU, self.waiting_cost]
        self.total_operating_cost = 0
        self.total_waiting_cost = 0
        
    def __str__(self):
        return "Name:" +self.name
            
    def berth_number(self, berth_number):
        """ The berth number the vessel is assigned to """
        self.allocated_berth = berth_number
    
    def show(self):
        print('Ship number: ', self.ship_number)
        print('Name: ', self.name)
        print('Arrival time: ', self.arrival_time)
        print('TEU: ', self.TEU)
        print('Operations time (QC days): ', self.operation_time)
        if self.allocated_berth == "-1": 
            print("Not assigned to a berth yet")
        else:
            print("Berth Number: ", self.allocated_berth)

def arrivalTime(oldTime):
    mu = math.log(m**2/math.sqrt(v+m**2))
    sigma = math.sqrt(math.log(v/(m**2)+1))
    interArrivalTime = np.random.lognormal(mu,sigma)
    arrivalTime = oldTime+interArrivalTime
    arrivalTime = round(arrivalTime,2)
    return arrivalTime
    
def waitingCost():
    cost = int(random.gauss(waiting_cost_mean, waiting_cost_stdev))
    return cost
        
def numberOfTEUs():
    TEU = random.randint(TEU_LB, TEU_UB)
    return TEU
    
def pickName():
    name = names['Names'][random.randint(0,len(names)-1)]
    return name

def createShips():
    ships = list()
    oldArrivalTime = 0
    for i in range(number_of_ships):
        name = pickName()
        TEUs = numberOfTEUs()
        arrival_time = arrivalTime(oldArrivalTime)
        oldArrivalTime = arrival_time
        waiting_cost = waitingCost()
        ships.append(Ship(i, name, arrival_time, TEUs, waiting_cost))
    return ships

def saveShipsToCSV(ships, filename):
    shipNames = list()
    shipsArrivalTime = list()
    shipsTEU = list()
    shipsWaitingcost = list()
    for ship in ships:
        shipNames.append(ship.name)
        shipsArrivalTime.append(ship.arrival_time)
        shipsTEU.append(ship.TEU)
        shipsWaitingcost.append(ship.waiting_cost)
    
    dataframe = pd.DataFrame(
                             {'Name': shipNames, 
                              'Arrival Time': shipsArrivalTime, 
                              'TEU': shipsTEU,
                              'Waiting Cost': shipsWaitingcost
                              })
    dataframe = dataframe[['Name', 'Arrival Time', 'TEU', 'Waiting Cost']]
    dataframe.to_csv(path+filename , index = False)
    
def play(set_number):
    filename = 'set_of_ships_{0}.csv'.format(set_number)
    ships = createShips()
    saveShipsToCSV(ships, filename)
    return ships

if __name__ == "__main__":
    ships = play(1)    

