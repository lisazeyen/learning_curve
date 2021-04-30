#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:45:41 2021

@author: bw0928
"""
import os, sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

os.chdir("/home/ws/bw0928/Dokumente/PyPSA")
sys.path = [os.pardir] + sys.path
import pypsa

print(pypsa.__file__)
#%%
def get_social_discount(t, r=0.01):
    return (1/(1+r)**t)


def set_new_sns_invp(n, inv_years):
    """
    set new snapshots (sns) for all time varying componentents and
    investment_periods depending on investment years ('inv_years')

    input:
        n: pypsa.Network()
        inv_years: list of investment periods, e.g. [2020, 2030, 2040]

    """

    for component in n.all_components:
        pnl = n.pnl(component)
        attrs = n.components[component]["attrs"]

        for k,default in attrs.default[attrs.varying].iteritems():
            pnl[k] = pd.concat([(pnl[k].rename(index=lambda x: x.replace(year=year), level=1)
                                       .rename(index=lambda x: n.snapshots.get_level_values(level=1)[0].replace(year=year), level=0))
                                for year in inv_years])

    # set new snapshots + investment period
    n.snapshot_weightings = pd.concat([(n.snapshot_weightings.rename(index=lambda x: x.replace(year=year), level=1)
                                       .rename(index=lambda x: n.snapshots.get_level_values(level=1)[0].replace(year=year), level=0))
                                for year in inv_years])
    n.set_snapshots(n.snapshot_weightings.index)
    n.set_investment_periods(n.snapshots)


def get_investment_weighting(energy_weighting, r=0.01):
    """
    returns cost weightings depending on the the energy_weighting (pd.Series)
    and the social discountrate r
    """
    end = energy_weighting.cumsum()
    start = energy_weighting.cumsum().shift().fillna(0)
    return pd.concat([start,end], axis=1).apply(lambda x: sum([get_social_discount(t,r)
                                                               for t in range(int(x[0]), int(x[1]))]),
                                                axis=1)


#%%
years = [2020, 2030, 2040, 2050]
investment = pd.DatetimeIndex(['{}-01-01 00:00'.format(year) for year in years])
r = 0.01 # social discountrate
path_eur = "/home/ws/bw0928/Dokumente/pypsa-eur/"
# Only consider a few snapshots to speed up the calculations
nhours=876

n = pypsa.Network(path_eur + "networks/elec_s_45_ec.nc")

# For GlobalConstraint of the technical limit at each node, get the p_nom_max
p_nom_max_limit = n.generators.p_nom_max.groupby([n.generators.carrier, n.generators.bus]).sum()

n.set_snapshots(n.snapshots[::nhours])
n.snapshot_weightings.loc[:] = nhours

snapshots = pd.MultiIndex.from_product([years, n.snapshots])
n.set_snapshots(snapshots)

sns=n.snapshots


# set investment period weightings
# last year is weighted by 1
n.investment_period_weightings.loc[:, "energy_weighting"] = n.investment_period_weightings.index.to_series().diff().shift(-1).fillna(1).values
# set investment_weighting
n.investment_period_weightings.loc[:, "objective_weightings"] = get_investment_weighting(n.investment_period_weightings["energy_weighting"], r)
n.investment_period_weightings.loc[:, "time_weightings"] = n.investment_period_weightings.index.to_series().diff().shift(-1).fillna(1).values

n.investment_period_weightings


# ### Play around with assumptions:
# 1. conventional phase out  <br>
# 2. renewable generators are build in every investment_period  <br>
# 3. build years for certain AC Lines or DC Links <br>
# 4. global constraints <br>
#     a. carbon budget  <br>
#     b. limit onshore wind and solar capacities at each node for each investment period

# 1) conventional phase out
# conventional lifetime + build year
conventionals = ["lignite", "coal", "oil", "nuclear", "CCGT", "OCGT"]
gens = n.generators[n.generators.carrier.isin(conventionals)].index
n.generators.loc[gens, "build_year"] = 2013
n.generators.loc[gens, "lifetime"] = 20


# 2.) renewable generator assumptions (e.g. can be newly build in each investment
# period, capital costs are decreasing,...)

# renewable
renewables = ["solar", "onwind", "offwind-ac", "offwind-dc"]
gen_names = n.generators[n.generators.carrier.isin(renewables)].index
df = n.generators.loc[gen_names]
p_max_pu = n.generators_t.p_max_pu[gen_names]

# drop old renewable generators
n.generators.drop(gen_names, inplace=True)
n.generators_t.p_max_pu.drop(gen_names, axis=1, inplace=True)

# add new renewable generator for each investment period
counter = 0
for year in years:
    n.madd("Generator",
           df.index,
           suffix=" " + str(year),
           bus=df.bus,
           carrier=df.carrier,
           p_nom_extendable=True,
           p_nom_max=df.p_nom_max,
           build_year=year,
           marginal_cost=df.marginal_cost,
           lifetime=15,
           capital_cost=df.capital_cost,
           efficiency=df.efficiency * 1.01**counter,
           p_max_pu=p_max_pu)

    counter += 1


n.generators[(n.generators.carrier=="solar") & (n.generators.bus=="DE0 0")]


# 3.) build year / transmission expansion for AC Lines and DC Links

# later_lines = n.lines.iloc[::5].index
# n.lines.loc[later_lines, "build_year"] = 2030
# later_lines = n.lines.iloc[::7].index
# n.lines.loc[later_lines, "build_year"] = 2040
# n.lines["s_nom_extendable"] = True
# n.links["p_nom_extendable"] = True


# 4.) Test global constraints <br>
# a) CO2 constraint: can be specified as a budget,  which limits the CO2
#    emissions over all investment_periods
# b)  or/and implement as a constraint for an investment period, if the
# GlobalConstraint attribute "investment_period" is not specified, the limit
#  applies for each investment period.


# (a) add CO2 Budget constraint
n.add("GlobalConstraint",
      "CO2Budget",
      type="Budget",
      carrier_attribute="co2_emissions", sense="<=",
      constant=1e7)

# add CO2 limit for last investment period
# n.add("GlobalConstraint",
#       "CO2Limit",
#       carrier_attribute="co2_emissions", sense="<=",
#       investment_period = sns.levels[0][-1],
#       constant=1e2)


# b) NOT WORKING YET add constraint for the technical maximum capacity for a carrier (e.g. "onwind", at one node and for one investment_period)

# global p_nom_max for each carrier + investment_period at each node
p_nom_max_inv_p = pd.DataFrame(np.repeat([p_nom_max_limit.values],
                                         len(sns.levels[0]), axis=0),
                               index=sns.levels[0], columns=p_nom_max_limit.index)

#n.add("GlobalConstraint",
#      "TechLimit",
#      carrier_attribute=["onwind", "solar"],
#      sense="<=",
#      type="tech_capacity_expansion_limit",
#      constant=p_nom_max_inv_p[["onwind", "solar"]])




n.global_constraints

n.carriers.loc["solar", "learning_rate"] = 0.4
n.carriers.loc["solar", "global_capacity"] = 1e9

#%%
n.lopf(pyomo=False, solver_name="gurobi", skip_objective=True,
       multi_investment_periods=True,
       extra_functionality=add_learning)