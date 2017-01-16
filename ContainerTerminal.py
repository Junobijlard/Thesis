#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 21:57:45 2016

@author: Juno
"""

import numpy as np
import pandas as pd
import random
import math

#==============================================================================
# PARAMETERS:
# seaport_type: either small, normal or large seaport
# number_of_terminals: number of terminals to create
# path: path of save file
# nameOfFile: name of save file
# names: names of ports
# berth_UB, berth_LB: upper/lower bound of number of berths in port
# QC_UB, QC_LB: upper/lower bound of number of QCs in port
#==============================================================================

seaport_type = 3 #small = 1, normal = 2, large = 3
number_of_terminals = 1
path = '/Users/Juno/Desktop/Scriptie/Python/Ship configurations/'
nameOfFile = 'Container Terminal configurations.csv'
pathOfNames = '/Users/Juno/Desktop/Scriptie/Python/Ship configurations/Terminal Names.csv'

if seaport_type == 1:
    seaport_type_name = "Small"
    berth_UB = 4
    berth_LB = 2
    QC_UB = 8
    QC_LB = 3
elif seaport_type == 2:
    seaport_type_name = "Normal"
    berth_UB = 8
    berth_LB = 5
    QC_UB = 16
    QC_LB = 6
else:
    seaport_type_name = "Large"
    berth_UB = 12
    berth_LB = 9
    QC_UB = 24
    QC_LB = 10
    
#==============================================================================
# CODE:
#==============================================================================
names = pd.read_csv(pathOfNames)
pathOfContainerFile = path+nameOfFile
class ContainerTerminal(object):
    """Class used to generate a container terminal with attributes: 
        index: index number of port
        seaport type: large, medium or small port
        berth_positions: number of berths in container terminal
        QC_number: number of QC's in port"""
    def __init__(self, name, seaport_type_name, berth_positions, QC_number, ship):
        self.name = name
        self.seaport_type = seaport_type_name
        self.berth_positions = berth_positions
        self.QC_number = QC_number
        self.berths = dict(zip(range(berth_positions),[ship]*berth_positions))
    
    def show(self):
        print('name: ', self.name)
        print('seaport_type: ', self.seaport_type)
        print('berth_positions: ', self.berth_positions)
        print('QC_number: ', self.QC_number)

def createTerminals():
    terminals = list()
    for i in range(number_of_terminals):
        name = names['Names'][random.randint(0,len(names)-1)]
        berth_positions = random.randint(berth_LB,berth_UB)
        QC_number = random.randint(QC_LB, QC_UB)
        terminals.append(ContainerTerminal(name, seaport_type_name, berth_positions, QC_number))
    return terminals

def saveContainerTerminalToCSV(terminals):
    terminalName = list()
    terminal_seaport_type = list()
    terminal_berth_positions = list()
    terminal_QC_number = list()
    
    for terminal in terminals:
        terminalName.append(terminal.name)
        terminal_seaport_type.append(terminal.seaport_type)
        terminal_berth_positions.append(terminal.berth_positions)
        terminal_QC_number.append(terminal.QC_number)
        
    df = pd.DataFrame({
                       'Name': terminalName,
                       'Type': terminal_seaport_type,
                       'Berth Positions': terminal_berth_positions,
                       'QC Number': terminal_QC_number})
    df = df[['Name', 'Type', 'Berth Positions', 'QC Number']]
    df.to_csv(pathOfContainerFile, index=False)

def play():
    terminals = createTerminals()
    saveContainerTerminalToCSV(terminals)

if __name__ == '__main__':
    play()
