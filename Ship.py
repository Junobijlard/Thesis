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
nameOfFile = 'ships_bayu.csv'

path = '/Users/Juno/Desktop/Scriptie/Python/Ship configurations/'
pathOfNames = '/Users/Juno/Desktop/Scriptie/Python/Ship configurations/Ship_names.csv'

#==============================================================================
# CODE
#==============================================================================

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
        self.arrival_time = round(arrival_time,2)
        self.numberOfTEUs = TEU
        operation = TEU*3/60/24
        self.operation_time = round(operation,2)
        self.waiting_cost = waiting_cost
        self.allocated_berth = "-1"
        self.assigned = False
        self.starting_time = None
        self.finishing_time = None
        self.training_values = [self.arrival_time, self.numberOfTEUs, self.waiting_cost]
        
    def __str__(self):
        return "Name:" +self.name
            
    def berth_number(self, berth_number):
        """ The berth number the vessel is assigned to """
        self.allocated_berth = berth_number
    
    def show(self):
        print('Ship number: ', self.ship_number)
        print('Name: ', self.name)
        print('Arrival time: ', self.arrival_time)
        print('TEU: ', self.numberOfTEUs)
        print('Operations time (QC days): ', self.operation_time)
        if self.allocated_berth == "-1": 
            print("Not assigned to a berth yet")
        else:
            print("Berth Number: ", self.allocated_berth)

def arrivalTime(arrival_number):
    mu = math.log(m**2/math.sqrt(v+m**2))
    sigma = math.sqrt(math.log(v/(m**2)+1))
    i = 0
    randomValue = 0
    while i <= arrival_number:
        randomValue+= np.random.lognormal(mu,sigma)
        i+=1
    randomValue = randomValue/24
    randomValue = round(randomValue, 2)
    return randomValue
        
def numberOfTEUs():
    TEU = random.randint(TEU_LB, TEU_UB)
    return TEU
    
def pickName():
    name = names['Names'][random.randint(0,len(names)-1)]
    return name

def createShips():
    ships = list()
    for i in range(number_of_ships):
        name = pickName()
        TEUs = numberOfTEUs()
        arrival_time = arrivalTime(i)
        ships.append(Ship(i, name, arrival_time, TEUs))
    return ships

def saveShipsToCSV(ships):
    shipNames = list()
    shipsArrivalTime = list()
    shipsTEU = list()
    for ship in ships:
        shipNames.append(ship.name)
        shipsArrivalTime.append(ship.arrival_time)
        shipsTEU.append(ship.numberOfTEUs)
    
    dataframe = pd.DataFrame(
                             {'Name': shipNames, 
                              'Arrival Time': shipsArrivalTime, 
                              'TEU': shipsTEU, 
                              })
    dataframe = dataframe[['Name', 'Arrival Time', 'TEU']]
    dataframe.to_csv(pathOfShipFile , index = False)
    
def play():
    ships = createShips()
    saveShipsToCSV(ships)
    return ships

if __name__ == "__main__":
    ships = play()    

