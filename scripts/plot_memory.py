#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 13:37:26 2021

@author: lisa
"""
import pandas as pd
import matplotlib.pyplot as plt

label = "Co2L-3h-learnH2xElectrolysisp0-local_sec"
# /home/lisa/mnt/lisa/learning_curve/results/split_regions_withretro2/logs/elec_s_EU_Co2L-3h-learnH2xElectrolysisp0-local_sec_memory.log
infile = r"/home/lisa/mnt/lisa/learning_curve/results/split_regions_withretro2/logs/elec_s_EU_{}_memory.log".format(label)

with open(infile) as f:
    f = f.readlines()

memory = [float(line.split(" ")[1]) for line in f]
time = [float(line.split(" ")[2][:-1]) for line in f]
time = [x- time[0] for x in time]

mem_series = pd.Series(memory, index=time)

#%%
ax = (mem_series/1e3).plot(grid=True, title="Memory usage")
ax.set_ylabel("memory \n [GB]")
ax.set_xlabel("time \n [s]")

plt.savefig(infile.split("logs/")[0] + "graphs/memory_{}.pdf".format(label),
            bbox_inches="tight")
