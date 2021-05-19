#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:45:41 2021

@author: bw0928
"""
import os
import pandas as pd
import numpy as np
import math
from six import iteritems, string_types
import re
import xarray as xr

import logging
from vresutils.benchmark import memory_logger
from vresutils.costdata import annuity

import pypsa_learning as pypsa
from learning import add_learning


print(pypsa.__file__)


if 'snakemake' not in globals():
    os.chdir("/home/ws/bw0928/Dokumente/learning_curve/scripts")
    from _helpers import mock_snakemake
    snakemake = mock_snakemake('solve_network', lv='1.0', sector_opts='Co2L-876H-learnsolarp0',clusters='37')

logger = logging.getLogger(__name__)
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



def average_every_nhours(n, offset):
    logger.info('Resampling the network to {}'.format(offset))
    m = n.copy(with_time=False)

    #fix copying of network attributes
    #copied from pypsa/io.py, should be in pypsa/components.py#Network.copy()
    allowed_types = (float,int,bool) + string_types + tuple(np.typeDict.values())
    attrs = dict((attr, getattr(n, attr))
                 for attr in dir(n)
                 if (not attr.startswith("__") and
                     isinstance(getattr(n,attr), allowed_types)))
    for k,v in iteritems(attrs):
        setattr(m,k,v)

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name+"_t")
        for k, df in iteritems(c.pnl):
            if not df.empty:
                if c.list_name == "stores" and k == "e_max_pu":
                    pnl[k] = df.resample(offset).min()
                elif c.list_name == "stores" and k == "e_min_pu":
                    pnl[k] = df.resample(offset).max()
                else:
                    pnl[k] = df.resample(offset).mean()

    return m


def prepare_network(n, solve_opts=None):
    if solve_opts is None:
        solve_opts = snakemake.config['solving']['options']

    if 'clip_p_max_pu' in solve_opts:
        for df in (n.generators_t.p_max_pu, n.generators_t.p_min_pu, n.storage_units_t.inflow):
            df.where(df>solve_opts['clip_p_max_pu'], other=0., inplace=True)

    if solve_opts.get('load_shedding'):
        n.add("Carrier", "Load")
        n.madd("Generator", n.buses.index, " load",
               bus=n.buses.index,
               carrier='load',
               sign=1e-3, # Adjust sign to measure p and p_nom in kW instead of MW
               marginal_cost=1e2, # Eur/kWh
               # intersect between macroeconomic and surveybased
               # willingness to pay
               # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
               p_nom=1e9 # kW
        )

    if solve_opts.get('noisy_costs'):
        for t in n.iterate_components():
            #if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if 'marginal_cost' in t.df:
                np.random.seed(174)
                t.df['marginal_cost'] += 1e-2 + 2e-3*(np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(['Line', 'Link']):
            np.random.seed(123)
            t.df['capital_cost'] += (1e-1 + 2e-2*(np.random.random(len(t.df)) - 0.5)) * t.df['length']

    if solve_opts.get('nhours'):
        nhours = solve_opts['nhours']
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760./nhours

    return n


def prepare_costs(cost_file, discount_rate, lifetime):
    """
    prepare cost data
    """
    #set all asset costs and other parameters
    costs = pd.read_csv(cost_file,index_col=list(range(2))).sort_index()

    #correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"),"value"]*=1e3

    #min_count=1 is important to generate NaNs which are then filled by fillna
    costs = costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    costs = costs.fillna({"CO2 intensity" : 0,
                          "FOM" : 0,
                          "VOM" : 0,
                          "discount rate" : discount_rate,
                          "efficiency" : 1,
                          "fuel" : 0,
                          "investment" : 0,
                          "lifetime" : lifetime
    })

    costs["fixed"] = [(annuity(v["lifetime"],v["discount rate"])+v["FOM"]/100.)*v["investment"] for i,v in costs.iterrows()]
    return costs

def prepare_costs_all_years(years):
    """
    prepares cost data for multiple years
    """
    all_costs = {}

    for year in years:
        all_costs[year] = prepare_costs(snakemake.input.costs + "/costs_{}.csv".format(year),
                              snakemake.config['costs']['discountrate'],
                              snakemake.config['costs']['lifetime'])
    return all_costs


def update_wind_solar_costs(n,costs, years):
    """
    Update costs for wind and solar generators added with pypsa-eur to those
    cost in the planning year

    """

    #assign clustered bus
    #map initial network -> simplified network
    busmap_s = pd.read_csv(snakemake.input.busmap_s, index_col=0).squeeze()
    busmap_s.index = busmap_s.index.astype(str)
    busmap_s = busmap_s.astype(str)
    #map simplified network -> clustered network
    busmap = pd.read_csv(snakemake.input.busmap, index_col=0).squeeze()
    busmap.index = busmap.index.astype(str)
    busmap = busmap.astype(str)
    #map initial network -> clustered network
    clustermaps = busmap_s.map(busmap)

    #NB: solar costs are also manipulated for rooftop
    #when distribution grid is inserted
    for year in years:
        n.generators.loc[(n.generators.carrier=='solar') & (n.generators.build_year==year),
                         'capital_cost'] = costs[year].at['solar-utility', 'fixed']

        n.generators.loc[(n.generators.carrier=='onwind') & (n.generators.build_year==year),
                         'capital_cost'] = costs[year].at['onwind', 'fixed']

        #for offshore wind, need to calculated connection costs
        #code adapted from pypsa-eur/scripts/add_electricity.py
        for connection in ['dc','ac']:
            tech = "offwind-" + connection
            profile = snakemake.input['profile_offwind_' + connection]
            with xr.open_dataset(profile) as ds:
                underwater_fraction = ds['underwater_fraction'].to_pandas()
                connection_cost = (snakemake.config['costs']['lines']['length_factor'] *
                                   ds['average_distance'].to_pandas() *
                                   (underwater_fraction *
                                    costs[year].at[tech + '-connection-submarine', 'fixed'] +
                                    (1. - underwater_fraction) *
                                    costs[year].at[tech + '-connection-underground', 'fixed']))

                #convert to aggregated clusters with weighting
                weight = ds['weight'].to_pandas()

                #e.g. clusters == 37m means that VRE generators are left
                #at clustering of simplified network, but that they are
                #connected to 37-node network
                if snakemake.wildcards.clusters[-1:] == "m":
                    genmap = busmap_s
                else:
                    genmap = clustermaps

                connection_cost = (connection_cost*weight).groupby(genmap).sum()/weight.groupby(genmap).sum()

                capital_cost = (costs[year].at['offwind', 'fixed'] +
                                costs[year].at[tech + '-station', 'fixed'] +
                                connection_cost)

                logger.info("Added connection cost of {:0.0f}-{:0.0f} Eur/MW/a to {}"
                            .format(connection_cost[0].min(), connection_cost[0].max(), tech))

                n.generators.loc[(n.generators.carrier==tech)  & (n.generators.build_year==year),
                                 'capital_cost'] = capital_cost.rename(index=lambda node: node + ' ' + tech + " " +str(year))

#%%
years = [2020, 2030, 2040, 2050]
investment = pd.DatetimeIndex(['{}-01-01 00:00'.format(year) for year in years])
r = 0.01 # social discountrate
# Only consider a few snapshots to speed up the calculations
n = pypsa.Network(snakemake.input.network)

# costs
costs = prepare_costs_all_years(years)

n.global_constraints.drop(n.global_constraints.index, inplace=True)
global_capacity = pd.read_csv(snakemake.input.global_capacity, index_col=0)

opts = snakemake.wildcards.sector_opts.split('-')

for o in opts:
    # learning
    if "learn" in o:
        techs = list(snakemake.config["learning_rates"].keys())
        if any([tech in o for tech in techs]):
            tech = [tech for tech in techs if tech in o][0]
            factor = float(o[len("learn"+tech):].replace("p",".").replace("m","-"))
            learning_rate = snakemake.config["learning_rates"][tech] + factor
            logger.info("technology learning for {} with learning rate {}%".format(tech, learning_rate*100))
            n.carriers.loc[tech, "learning_rate"] = learning_rate
            n.carriers.loc[tech, "global_capacity"] = global_capacity.loc[tech, 'Electricity Installed Capacity (MW)']
            n.carriers.loc[tech, "max_capacity"] = 4 * global_capacity.loc[tech, 'Electricity Installed Capacity (MW)']

    # temporal clustering
    m = re.match(r'^\d+h$', o, re.IGNORECASE)
    if m is not None:
        n = average_every_nhours(n, m.group(0))
    # representive snapshots
    m = re.match(r'^\d+sn$', o, re.IGNORECASE)
    if m is not None:
        sn = int(m.group(0).split("sn")[0])
        n.set_snapshots(n.snapshots[::sn])
        n.snapshot_weightings *= sn

# For GlobalConstraint of the technical limit at each node, get the p_nom_max
p_nom_max_limit = n.generators.p_nom_max.groupby([n.generators.carrier, n.generators.bus]).sum()


snapshots = pd.MultiIndex.from_product([years, n.snapshots])
n.set_snapshots(snapshots)

sns=n.snapshots

# set investment_weighting
n.investment_period_weightings.loc[:, "time_weightings"] = n.investment_period_weightings.index.to_series().diff().shift(-1).fillna(1).values
n.investment_period_weightings.loc[:, "objective_weightings"] = get_investment_weighting(n.investment_period_weightings["time_weightings"], r)


# ### Play around with assumptions:


# 1) conventional phase out
# conventional lifetime + build year
conventionals = ["lignite", "coal", "oil", "nuclear", "CCGT","OCGT"]
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
for year in years:
    df.lifetime = costs[year].loc[df.carrier.apply(lambda x: x.split("-")[0]), "lifetime"].values
    n.madd("Generator",
           df.index,
           suffix=" " + str(year),
           bus=df.bus,
           carrier=df.carrier,
           p_nom_extendable=True,
           p_nom_max=df.p_nom_max,
           build_year=year,
           marginal_cost=df.marginal_cost,
           lifetime=df.lifetime,
           capital_cost=df.capital_cost,
           efficiency=df.efficiency,
           p_max_pu=p_max_pu)


update_wind_solar_costs(n,costs, years)

df = n.generators[n.generators.carrier=="OCGT"]
# add OCGT
for year in years:
    n.madd("Generator",
           df.index,
           suffix=" " + str(year),
           bus=df.bus,
           carrier=df.carrier,
           p_nom_extendable=True,
           build_year=year,
           marginal_cost=df.marginal_cost,
           lifetime=costs[year].loc["OCGT", "lifetime"],
           capital_cost=costs[year].loc["OCGT", "fixed"],
           efficiency=costs[year].loc["OCGT", "efficiency"]
           )

for tech in n.carriers[n.carriers.learning_rate!=0].index:
    n.carriers.loc[tech, "initial_cost"] = n.generators.loc[(n.generators.carrier==tech) & (n.generators.build_year==years[0]),
                             'capital_cost'].mean()




# increase load
weight = pd.Series(np.arange(1.,2.0, 1/len(years)), index=years)
n.loads_t.p_set = n.loads_t.p_set.mul(weight, level=0, axis="index")
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
      constant=1e8)

n.add("GlobalConstraint",
      "CO2neutral",
      type="primary_energy",
      carrier_attribute="co2_emissions", sense="<=",
      constant=0)
# add CO2 limit for last investment period
# n.add("GlobalConstraint",
#       "CO2Limit",
#       carrier_attribute="co2_emissions", sense="<=",
#       investment_period = sns.levels[0][-1],
#       constant=1e2)




# global p_nom_max for each carrier + investment_period at each node
p_nom_max_inv_p = pd.DataFrame(np.repeat([p_nom_max_limit.values],
                                         len(sns.levels[0]), axis=0),
                               index=sns.levels[0], columns=p_nom_max_limit.index)

for carrier in renewables:
    nodes = p_nom_max_inv_p[carrier].columns
    max_cap = p_nom_max_inv_p[carrier].iloc[0,:].rename(lambda x: "TechLimit " + x + " " +carrier)
    n.madd("GlobalConstraint",
          "TechLimit " + nodes + " " + carrier,
          carrier_attribute=carrier,
          sense="<=",
          type="tech_capacity_expansion_limit",
          bus=nodes,
          constant=max_cap)




if any(n.carriers.learning_rate!=0):
    extra_functionality = add_learning
    skip_objective = True
else:
    extra_functionality = None
    skip_objective = False

config = snakemake.config['solving']
solve_opts = config['options']
solver_options = config['solver'].copy()
solver_log = snakemake.log.solver

# MIPFocus = 3, MIPGap, MIRCuts=2 (agressive)

logging.basicConfig(filename=snakemake.log.python,
                    level=snakemake.config['logging']['level'])

with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

    n = prepare_network(n, solve_opts)
    n.lopf(pyomo=False, solver_name="gurobi", skip_objective=skip_objective,
           multi_investment_periods=True, solver_options=solver_options,
           solver_logfile=solver_log, keep_files=True,
           extra_functionality=extra_functionality, keep_shadowprices=False)

    n.export_to_netcdf(snakemake.output[0])

logger.info("Maximum memory usage: {}".format(mem.mem_usage))

