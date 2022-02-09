#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 13:37:26 2021

@author: lisa
"""
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if "snakemake" not in globals():
        import os

        # os.chdir("/home/lisa/Documents/learning_curve/scripts")
        os.chdir("/home/lisa/mnt/lisa/learning_curve/scripts")
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_memory",
            sector_opts="Co2L-25sn-notarget-1p6-learnH2xElectrolysisp0-learnsolarp0-learnonwindp0",
            clusters="37",
        )


with open(snakemake.input.log) as f:
    f = f.readlines()

memory = [float(line.split(" ")[1]) for line in f]
time = [float(line.split(" ")[2][:-1]) for line in f]
time = [(x- time[0])/(60*60) for x in time]

mem_series = pd.Series(memory, index=time)

mem_series.rename(index=lambda x: float(x),inplace=True)

ax = (mem_series/1e3).plot(grid=True, title="Memory usage")
ax.set_ylabel("memory \n [GB]")
ax.set_xlabel("time \n [hours]")

plt.savefig(snakemake.output.memory_plot,
            bbox_inches="tight")