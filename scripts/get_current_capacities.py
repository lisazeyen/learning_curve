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
    os.chdir("/home/ws/bw0928/Dokumente/learning_curve/scripts")
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

# ----------------------------------------------------------------------------
# LOCAL CAPCAITIES ###########################################################
index = pd.Index(irena_map.values()).unique()
# capacities of chosen countries
cap_local = capacities[(capacities.Region=="Europe") & (capacities.alpha_2.isin(countries))]
cap_local = (cap_local[cap_local.carrier.isin(index)]['Electricity Installed Capacity (MW)']
             .groupby([cap_local.carrier, cap_local.alpha_2]).sum())
cap_local.to_csv(snakemake.output.local_capacities)

# ----------------------------------------------------------------------------
# GLOBAL CAPACITIES ##########################################################
# IRENA onwind, solar, offwind, nuclear
global_caps = capacities['Electricity Installed Capacity (MW)'].groupby(capacities.carrier).sum()

# H2 electrolysis ################################
# https://www.researchgate.net/publication/321682272_Future_cost_and_performance_of_water_electrolysis_An_expert_elicitation_study
# appendix figure B.1
# IRENA 20 GW https://irena.org/-/media/Files/IRENA/Agency/Publication/2020/Dec/IRENA_Green_hydrogen_cost_2020.pdf
# check https://www.iea.org/reports/hydrogen looks much less 2020 100 MW/year
global_caps.loc["H2 electrolysis"] = 20*1e3  # figure in units GW_el
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
global_caps.loc['DAC'] = 1  # TODO
index = index.union(['DAC'])

# save global capacities ####################################
global_caps.loc[index].to_csv(snakemake.output.global_capacities)

# ----------------------------------------------------------------------------
# get fraction of global capacities
fraction = cap_local.groupby(level=0).sum() / global_caps.loc[index]
fraction.name = "Fraction of global installed capacity"
fraction.to_csv(snakemake.output.fraction)


