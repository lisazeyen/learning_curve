#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:41:18 2021

@author: bw0928
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('default')
fig, ax = plt.subplots()

y.index = x["solar"].values
y.rename(columns={"solar": "learning curve"}, inplace=True)
y_cum.index = x["solar"].values
y_cum.rename(columns={"solar": "cumulative cost curve"}, inplace=True)

plt.step(points["solar"]["x_fit"].shift(-1).dropna(), slope["solar"],
         color='#1f77b4', lw=2, ls="--")
y.plot(ax=ax, color="red", lw=3)
ax.set_ylabel("specific cost \n [Eur/kW]")
ax.set_xlabel("installed capacity \n [kW]")


ax2=ax.twinx()
(y_cum/1e3).plot(ax=ax2, color="olive", lw=3)
ax2.set_ylabel("cumulative cost \n [thousand Eur]")

lin = points["solar"].set_index("x_fit").rename(columns={"y_fit":"piece-wise linearisation"})
(lin/1e3).plot(marker="*", ls="--", ax=ax2, grid=True, lw=2, markersize=10)

plt.title(label="Learning rate {} -- c0 = {} Eur/kW -- e0 = {} kW".format(learning_rate, c0, e0))
plt.savefig("/home/ws/bw0928/Dokumente/learning_curve/graphics/linearisation.pdf",
            bbox_inches="tight")
#%%
# initial experience / initial capacity
e0 = 1
# cost per unit at e0
c0 = 1000
# learning rate cost reduction with every doubling of experience
learning_rate = 0.2
rates = np.arange(0,1,0.1)
x_low = e0
x_high= 1e2
# x position on learning curve = cumulative capacity
x = pd.DataFrame(np.linspace(x_low, x_high, 1000))
x = pd.DataFrame(np.arange(x_low, x_high, 1))

# interpolation points
points = pd.DataFrame(columns=pd.MultiIndex.from_product([rates, ["x_fit", "y_fit"]]), index=np.arange(segments))
y = pd.DataFrame(columns=rates, index=x.index)
y_cum = pd.DataFrame(columns=rates, index=x.index)
for lr in rates:

    y[lr] = x.apply(lambda x: experience_curve(x, lr, c0, e0))
    y_cum[lr] = x.apply(lambda x: cumulative_cost_curve(x, lr, c0, e0), axis=1)
    # get interpolation points
    points[lr] = piecewise_linear(y_cum[lr], segments, lr, c0, e0, 0)[0]

y.rename(columns=lambda x: round(x, ndigits=2), inplace=True)
y_cum.rename(columns=lambda x: round(x, ndigits=2), inplace=True)
points.rename(columns=lambda x: round(x, ndigits=2),level=0, inplace=True)
#%%
fig, ax = plt.subplots()

y.index = x[0].values
y.rename(columns={0: "learning curve"}, inplace=True)
y_cum.index = x[0].values
y_cum.rename(columns={0: "cumulative cost curve"}, inplace=True)

y.plot(ax=ax, lw=2, legend=False)
ax.set_ylabel("specific cost \n [Eur/kW]")
ax.set_xlabel("installed capacity \n [kW]")

ax2=ax.twinx()
(y_cum).plot(ax=ax2,  lw=2)
ax2.set_ylabel("cumulative cost \n [million Eur]")

for lr in points.columns.levels[0]:
    lin = points[lr].set_index("x_fit").rename(columns={"y_fit":"piece-wise linearisation"})
    lin.plot(marker="*", ls="--", ax=ax2, grid=True, lw=2, markersize=10)
plt.legend(bbox_to_anchor=(1.2,1))
#%%
lr=0.2
fig, ax = plt.subplots()

y.index = x[0].values
y.rename(columns={0: "learning curve"}, inplace=True)
y_cum.index = x[0].values
y_cum.rename(columns={0: "cumulative cost curve"}, inplace=True)

y[lr].plot(ax=ax, lw=2, legend=False)
ax.set_ylabel("specific cost \n [Eur/kW]")
ax.set_xlabel("installed capacity \n [kW]")

ax2=ax.twinx()
(y_cum[lr]).plot(ax=ax2,  lw=2)
ax2.set_ylabel("cumulative cost \n [million Eur]")


lin = points[lr].set_index("x_fit").rename(columns={"y_fit":"piece-wise linearisation"})
lin.plot(marker="*", ls="--", ax=ax2, grid=True, lw=2, markersize=10)
plt.legend(bbox_to_anchor=(1.2,1))