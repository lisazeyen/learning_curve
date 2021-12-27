#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:05:38 2021

data sources:
    - Today installed onshore, offshore and solar PV capacities
      from IRENA - Renewable Capacity Statistics (2020)
      https://irena.org/publications/2020/Mar/Renewable-Capacity-Statistics-2020

     - global battery capacities check
     JRC report "Li-ion batteries for mobility and stationary storage applications"
     p.16

reads today's installed capacities
@author: bw0928
"""
import pandas as pd
import pycountry

from distutils.version import LooseVersion
pd_version = LooseVersion(pd.__version__)
agg_group_kwargs = dict(numeric_only=False) if pd_version >= "1.3" else {}

#%%
irena_map = {'On-grid Solar photovoltaic': "solar",
             'Off-grid Solar photovoltaic': "solar",
             'Onshore wind energy': "onwind",
             'Offshore wind energy': "offwind",
             'Nuclear': "nuclear",}
#%%

if 'snakemake' not in globals():
    from vresutils.snakemake import MockSnakemake
    import yaml
    import os
    os.chdir("/home/lisa/Documents/learning_curve/scripts")
    snakemake = MockSnakemake(
        input=dict(IRENA= "data/IRENA_Stats_Tool_data.csv",),
        output=dict(global_capacities= "data/global_capacities.csv",
                    local_capacities="data/local_capacities.csv",
                    fraction="data/global_fraction.csv",
                    ))
    with open('../config.yaml', encoding='utf8') as f:
        snakemake.config = yaml.safe_load(f)

# considered countries
countries = snakemake.config["countries"]
# currently installed capacities
capacities = pd.read_csv(snakemake.input.IRENA, skiprows=[0,1,2,3,4, 6,7], usecols=[0,1,2,3,4,5,6,7,8])

# convert 3 letter iso code -> 2 letter iso code
iso_codes = capacities["ISO Code"].apply(lambda x:pycountry.countries.get(alpha_3=x)).dropna()
alpha2 = iso_codes.apply(lambda x: x.alpha_2)
capacities["alpha_2"] = alpha2

# add 2 letter iso of Kosovo
capacities.loc[capacities[capacities.Country=="Kosovo*"].index, "alpha_2"] = "KO"

# map IRENA carrier name to PyPSA syntax
capacities["carrier"] = capacities.Technology.replace(irena_map)

# convert capacity column to type float
a = capacities['Electricity Installed Capacity (MW)'].str.replace(",","", regex=True)
a = pd.to_numeric(a, errors="coerce")
capacities['Electricity Installed Capacity (MW)'] = a

# take data from 2020
capacities = capacities[capacities.Year == 2020]
# ----------------------------------------------------------------------------
# LOCAL CAPCAITIES ###########################################################
index = pd.Index(irena_map.values()).unique()
# capacities of chosen countries
cap_local = capacities[(capacities.Region=="Europe") & (capacities.alpha_2.isin(countries))]
cap_local = (cap_local[cap_local.carrier.isin(index)]['Electricity Installed Capacity (MW)']
             .groupby([cap_local.carrier, cap_local.alpha_2]).sum(**agg_group_kwargs))
cap_local.to_csv(snakemake.output.local_capacities)

# ----------------------------------------------------------------------------
# GLOBAL CAPACITIES ##########################################################
# IRENA onwind, solar, offwind, nuclear
global_caps = capacities['Electricity Installed Capacity (MW)'].groupby(capacities.carrier).sum(**agg_group_kwargs)

# H2 electrolysis ################################
# https://www.researchgate.net/publication/321682272_Future_cost_and_performance_of_water_electrolysis_An_expert_elicitation_study
# appendix figure B.1
# IRENA 20 GW https://irena.org/-/media/Files/IRENA/Agency/Publication/2020/Dec/IRENA_Green_hydrogen_cost_2020.pdf
# check https://www.iea.org/reports/hydrogen much less 2020 300 MW in 2020
# "Electrolysers have reached enough maturity to scale up manufacturing and
# deployment to significantly reduce costs, which is reflected in three
# consecutive years of record capacity deployment in 2018, 2019 and 2020.
# Despite the impact of the Covid‑19 pandemic, which has delayed a significant
# number of projects, close to 70 MW of electrolysis became operational in 2020,
# bringing total installed capacity to almost 300 MW. Europe has 40% of global
# installed capacity and will remain the dominant region thanks to the stimulus
# of policy support from numerous hydrogen strategies adopted in the last year
# and the prominence of electrolytic hydrogen in the Covid‑19 recovery packages
# of countries such as Germany, France and Spain."
# https://www.iea.org/data-and-statistics/data-product/hydrogen-projects-database

global_caps.loc["H2 electrolysis"] = 300
index = index.union(["H2 electrolysis"])

# H2 fuel cell #################################################
# figure 12 https://pubs.rsc.org/en/content/articlelanding/2019/ee/c8ee01157e#!divAbstract
global_caps.loc['H2 fuel cell'] = 1*1e3  # in 2015, figure in units GW
index = index.union(['H2 fuel cell'])

# battery storage ###########################################
# DEA technology data for energy storage, p.178 figure 9 global production capacity 174 GWh
# learning rate 18%
# IEA chart https://www.iea.org/data-and-statistics/charts/price-and-installed-capacity-of-li-ion-batteries-2010-2017
global_caps.loc["battery"] = 400 * 1e3  # in 2016 in GWh
index = index.union(["battery"])
# battery inverter #############################################
# DEA technology data for power storage, p.179 figure 10 for utility scale
global_caps.loc["battery inverter"] = 7000  # in 2020 in MW
index = index.union(["battery inverter"])

# DAC
# not really employed on a large scale
# https://www.iea.org/reports/direct-air-capture cite:
    # "Fifteen direct air capture plants are currently operational in Europe,
    # the United States and Canada. Most of these plants are small and sell the
    # captured CO2 for use – for carbonating drinks, for example."
global_caps.loc['DAC'] = 10  # TODO
index = index.union(['DAC'])

# save global capacities ####################################
global_caps.loc[index].to_csv(snakemake.output.global_capacities)

# ----------------------------------------------------------------------------
# get fraction of global capacities
fraction = cap_local.groupby(level=0).sum(**agg_group_kwargs) / global_caps.loc[index]
fraction.name = "Fraction of global installed capacity"
fraction.to_csv(snakemake.output.fraction)
