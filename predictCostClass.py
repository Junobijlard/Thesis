#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:36:22 2017

@author: juno
"""

class predictCostClass(object):
    def __init__(self, ship, v, starting_time, operations_cost):
        self.ship = ship
        self.v = v
        self.starting_time = starting_time
        self.arrival_time = ship.arrival_time
        self.finishing_time = self.starting_time + self.ship.operation_time/v
        self.operations_cost = operations_cost
        self.calculateCosts()
        
    def calculateCosts(self):
        self.waiting_cost = (self.finishing_time - self.arrival_time)*self.ship.waiting_cost
        self.operating_cost = (self.finishing_time - self.starting_time)*self.v*self.operations_cost
        self.total_cost = self.waiting_cost+self.operating_cost
        
    
