#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:07:20 2021
test SOS2 constraint gurobi
@author: lisa
"""
from gurobipy import Model, GRB, quicksum
from pypsa_learning.learning import get_linear_interpolation_points


# ##### PARAMETERS AND VARIABLES ##########################################
# get carriers with learning
learn_i = n.carriers[n.carriers.learning_rate != 0].index
segments = 5

# bounds for cumulative capacity -----------------------------------------
x_low = n.carriers.loc[learn_i, "global_capacity"]
x_high = n.carriers.loc[learn_i, "max_capacity"]


# ######## PIECEWIESE LINEARISATION #######################################
# get interpolation points (number of points = line segments + 1)
points = get_linear_interpolation_points(n, x_low, x_high, segments)

y_low = points["solar"]["y_fit"].min()
y_high = points["solar"]["y_fit"].max()
x_low = n.carriers.loc[learn_i, "global_capacity"].loc["solar"]
x_high = n.carriers.loc[learn_i, "max_capacity"].loc["solar"]
######################################################################
m = Model("test_SOS2")
# installed cumulative capacity
x = m.addVar(lb=x_low, ub=x_high, vtype=GRB.CONTINUOUS)
# total costs for cumulative capacity
y = m.addVar(lb=y_low, ub=y_high, vtype=GRB.CONTINUOUS)
# weights
weights = m.addVars(len(points), lb=0, ub=1, vtype=GRB.CONTINUOUS)
##########################################################################
demand_c = m.addConstr(y >= y_low*1.1 )
sos2 = m.addSOS(GRB.SOS_TYPE2, weights)
weight_sum = m.addConstr(quicksum(weights) == 1)
x_link = m.addConstr(quicksum([weights[i]*points["solar"]["x_fit"][i] for i in range(len(points))]) == x)
y_link =  m.addConstr(quicksum([weights[i]*points["solar"]["y_fit"][i] for i in range(len(points))]) == y)
################################################################
# Set the objective
obj = m.setObjective(y, GRB.MINIMIZE)
#################################################################
m.optimize()
m.write("/home/lisa/Documents/learning_curve/mip_start/test.lp")
m.write("/home/lisa/Documents/learning_curve/mip_start/test.mps")
