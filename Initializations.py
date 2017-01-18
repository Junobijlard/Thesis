#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 22:07:31 2016

@author: Juno
"""
from os import chdir
chdir("/Users/Juno/Desktop/Scriptie/Python")
import pandas as pd
from importlib import import_module
from Ship import Ship
from ContainerTerminal import ContainerTerminal

#Delfzijl containter terminal toegevoegd
k_k = list()
x_k = list()
y_k = list()
z_k = list()
j_k = list()
v_k = list()
ships = list()
terminals = list()

def read_data(module_name):
    mymod = import_module(module_name)
    path = getattr(mymod, "path")
    nameOfFile = getattr(mymod, "nameOfFile")
    totalPath = path + nameOfFile
    df = pd.read_csv(totalPath)
    return df

pd.set_option('precision', 2)
shipsDF = read_data("Ship")
containerTerminalsDF = read_data("ContainerTerminal")
dummy_ship = Ship(0, "dummy", 0,0,0)

for i in range(len(shipsDF)):
    ship = Ship(i, shipsDF['Name'][i], shipsDF['Arrival Time'][i], shipsDF['TEU'][i])
    ships.append(ship)

for i in range(len(containerTerminalsDF)):
    terminals.append(ContainerTerminal(containerTerminalsDF.iloc[i][0],containerTerminalsDF.iloc[i][1],
                                       containerTerminalsDF.iloc[i][2],containerTerminalsDF.iloc[i][3], dummy_ship))

delfzijl = ContainerTerminal('Delfzijl', 'Small', 2, 7, dummy_ship)
terminals.append(delfzijl)

def makeListOfNumbers(berthOrQC, terminal_number = -1):
    list_name = list()
    if berthOrQC=='berth':
        attribute = terminals[terminal_number].berth_positions
    else:
        attribute = terminals[terminal_number].QC_number
    for i in range(attribute):
        list_name.append(i)
    return list_name
    
QCs = makeListOfNumbers('QC')
berths = makeListOfNumbers('berth')

def initXYZ(berths):
    variable = dict(zip(berths, [0]*len(berths)))
    return variable

def createBerthDictionary():
    dictionary = {berth:[] for berth in berths}
    return dictionary
    
v,x,y,z = initXYZ(berths), initXYZ(berths), initXYZ(berths),initXYZ(berths)

v_k.append(v)
x_k.append(x)
y_k.append(y)
z_k.append(z)
j_k.append(0)
k_k.append(0)
berthDict = createBerthDictionary()
