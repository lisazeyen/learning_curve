#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:24:03 2021

@author: bw0928
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# Functions -------------------------------------------------------------------
def experience_curve(E, learning_rate, c0, e0=1):
    """
    calculates the specific investment costs c

    equations:

        (1) c = c0 / (E/e0)**alpha

        (2) learning_rate = 1 - 1/2**alpha



    Input
    ---------
    E            : cumulative experience
    learning_rate: Learning rate
    e0           : start experience, constant
    c0           : initial costs, constant
    """
    # calculate alpha
    alpha = math.log10(1 / (1-learning_rate)) / math.log10(2)

    # get specific investment
    c = c0 / (E/e0)**alpha

    return c


def production_costs(c0, progress_rate, p, p0):
    """
    calcultes production costs c depending on progress rate with the binary log
    lb:

        c = c0 * (p/p0)^(lb(progress_rate))
          = c0 * progress_rate^(lb(p/p0))

          because ln(a^(ln(b))) = ln(a) * ln(b) = ln(b^(ln(a)))

    Input:
    ---------------
        c0 :costs at time 0
        progress_rate : (1 - learning rate)
        p : cumulative production
        p0 : cummulative production at time 0
    """
    alpha = math.log(p/p0, 2)
    c = c0 * progress_rate**alpha

    return c

def get_production(c0, c, progress_rate):
    """
    calcultes cummulative production (p/p0)
    p/p0 = (c/c0) ** (1/)
    Input
    ---------------
    c0 :costs at time 0
    progress_rate : (1 - learning rate)
    """
    f = math.log(progress_rate,2)
    production = (c/c0)**(1/f)   #

    return production


def sigmoid_function(t):
    """
    s-shape function
    """
    return (1 / (1 + np.exp(-t)))


# MAIN -----------------------------------------------------------------------
costs = pd.read_csv("/home/lisa/Documents/technology-data/outputs/costs_2020.csv",
                    index_col=[0,1])

tech = "solar-utility"
c0 = costs.loc[(tech, "investment"), "value"]
e0 = 1 # 137.2 *1e6  # installed PV capacity in Europe in 2020
learning_rates = np.arange(0.05, 0.35, 0.03)
E = pd.Series(np.arange(1, 1e4, 100))
total = pd.concat([E.apply(lambda x: experience_curve(x, learning_rate, c0, e0)).rename(round(learning_rate, ndigits=2))
                   for learning_rate in learning_rates], axis=1)
total.index = E

total.plot(grid=True)
plt.xscale("log")
plt.yscale("log")
plt.title(tech)
plt.legend(title="Learning rate", loc="upper right")
plt.ylabel("specific investment costs")
plt.xlabel("cummulative installed capacity")
#%%
# sigmoid
t = pd.Series(np.arange(-6, 6, 0.1))
sigmoid =  t.apply(lambda x: sigmoid_function(x))
sigmoid.index = t
# linear approximation
linear_points = pd.Series([-6, -3, 0, 3,6])
linear_approx = linear_points.apply(lambda x: sigmoid_function(x))
linear_approx.index = linear_points
# plot
ax = sigmoid.plot(title="Sigmoid Function", lw=3)
linear_approx.plot(style="--o", ax=ax, color="green")
plt.grid(True)
plt.ylabel("y")
plt.xlabel("t")
plt.axvline(x=0, color="gray")
plt.axhline(y=0, color="gray")
# x is a variable which needs to be solved and y another variable which depends
plt.plot([x.X], [y.X], marker="*", markersize=12, color="r")
# non-linear on x
# %%
# from https://medium.com/bcggamma/hands-on-modeling-non-linearity-in-linear-optimization-problems-f9da34c23c9a
# optimise with gurobi
from gurobipy import Model, GRB, quicksum

# 0) Generate the data
# x limits
x_low = -6
x_high = 6
# x and y samples
x_samples = np.linspace(x_low, x_high, 5)
y_samples = pd.Series(x_samples).apply(sigmoid_function).values

# 1) Instantiate a new model
m = Model("test_MIP")

# 2) Declare the required variables
x = m.addVar(lb=x_low, ub=x_high, vtype=GRB.CONTINUOUS)
y = m.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
weights = m.addVars(len(x_samples), lb=0, ub=1, vtype=GRB.CONTINUOUS)

# 3) Define the set of weights and link them to x and y
sos2 = m.addSOS(GRB.SOS_TYPE2, weights)
weight_sum = m.addConstr(quicksum(weights) == 1)
x_link = m.addConstr(quicksum([weights[i]*x_samples[i] for i in range(len(x_samples))]) == x)
y_link = m.addConstr(quicksum([weights[i]*y_samples[i] for i in range(len(y_samples))]) == y)

y_cons = m.addConstr(-2.5<=x)
# 4) Set the objective
# obj = m.setObjective(y, GRB.MAXIMIZE)
obj = m.setObjective(y, GRB.MINIMIZE)
# 5) Optimize
m.optimize()
m.write("/home/lisa/Documents/learning_curve/mip_start/test_sos2.lp")
m.write("/home/lisa/Documents/learning_curve/mip_start/test_sos2.mps")

# 6) Print the results: the optimal value for variable x
print("Optimal value for x is: {}".format(x.X))
#%%
import gurobipy
from gurobipy import GRB
from pypsa_learning.linopt import set_int_index

problem_fn = "/home/lisa/Documents/learning_curve/mip_start/test.lp"
problem_fn = "/home/lisa/Documents/learning_curve/mip_start/pypsa-problem-at5rxsrw.lp"
problem_fn = "/tmp/pypsa-problem-08iqvbzr.lp"
m = gurobipy.read(problem_fn)
logging.disable(50)
for key, value in solver_options.items():
    m.setParam(key, value)
m.setParam("DualReductions", 0)
m.optimize()
m.computeIIS()
m.write("/home/lisa/Documents/learning_curve/mip_start/debug.ilp")
variables_sol = pd.Series({v.VarName: v.x for v in m.getVars()}).pipe(set_int_index)
try:
    constraints_dual = pd.Series({c.ConstrName: c.Pi for c in m.getConstrs()}).pipe(
        set_int_index
    )
except AttributeError:
    print("---\nShadow prices of MILP couldn't be parsed\n ---")
    constraints_dual = pd.Series(index=[c.ConstrName for c in m.getConstrs()])
objective = m.ObjVal
#%%
ax=points.droplevel(level=0,axis=1).set_index("x_fit").plot(marker="o",
                                                         markersize=10)
lösung.columns = ["x", "y"]

lösung.set_index("x").plot(ax=ax, marker="s", lw=0, markersize=12)
