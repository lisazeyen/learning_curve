#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:45:41 2021

@author: bw0928
"""
import os
import pandas as pd
import numpy as np
from six import iteritems, string_types, iterkeys
import re

import logging
from vresutils.benchmark import memory_logger
from vresutils.costdata import annuity

import pypsa_learning as pypsa
from pypsa_learning.learning import add_learning
from pypsa_learning.temporal_clustering import aggregate_snapshots
from pypsa_learning.descriptors import nominal_attrs, get_extendable_i, get_active_assets, expand_series

from pypsa_learning.linopt import get_var, linexpr, define_constraints
from pypsa_learning.io import import_components_from_dataframe


logger = logging.getLogger(__name__)


#First tell PyPSA that links can have multiple outputs by
#overriding the component_attrs. This can be done for
#as many buses as you need with format busi for i = 2,3,4,5,....
#See https://pypsa.org/doc/components.html#link-with-multiple-outputs-or-inputs
override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
override_component_attrs["Link"].loc["bus3"] = ["string",np.nan,np.nan,"3rd bus","Input (optional)"]
override_component_attrs["Link"].loc["bus4"] = ["string",np.nan,np.nan,"4th bus","Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["efficiency3"] = ["static or series","per unit",1.,"3rd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["efficiency4"] = ["static or series","per unit",1.,"4th bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]
override_component_attrs["Link"].loc["p3"] = ["series","MW",0.,"3rd bus output","Output"]
override_component_attrs["Link"].loc["p4"] = ["series","MW",0.,"4th bus output","Output"]

#%%
# maps pypsa name to technology data name
map_dict = {'H2 electrolysis':"electrolysis",
            'H2 fuel cell': "fuel cell",
            'battery charger':"battery inverter",
            # 'battery discharger':"battery inverter",
            "H2" : 'hydrogen storage underground',
            "battery": "battery storage",
            "offwind-ac": "offwind",
            "offwind-dc":"offwind",
            "solar rooftop": "solar-rooftop",
            "solar": "solar-utility",
            "DAC": "direct air capture"}

opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}

def get_social_discount(t, r=0.01):
    """Calculate for a given time t the social discount."""
    return (1/(1+r)**t)


def get_investment_weighting(energy_weighting, r=0.01):
    """Define cost weighting.

    Returns cost weightings depending on the the energy_weighting (pd.Series)
    and the social discountrate r
    """
    end = energy_weighting.cumsum()
    start = energy_weighting.cumsum().shift().fillna(0)
    return pd.concat([start,end], axis=1).apply(lambda x: sum([get_social_discount(t,r)
                                                               for t in range(int(x[0]), int(x[1]))]),
                                                axis=1)


def average_every_nhours(n, offset):
    """Temporal aggregate pypsa Network depending on offset."""
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
    """Add solving options to the network before calling the lopf.

    Solving options (solve_opts (type dict)), keys which effect the network in
    this function:

         'load_shedding'
         'noisy_costs'
         'clip_p_max_pu'
         'n_hours'
    """
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
    """Prepare cost data."""
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

def prepare_costs_all_years(years, update=False):
    """Prepare cost data for multiple years."""
    all_costs = {}
    for year in years:
        all_costs[year] = prepare_costs(snakemake.input.costs + "/costs_{}.csv".format(year),
                              snakemake.config['costs']['discountrate'],
                              snakemake.config['costs']['lifetime'])

    if not update:
        for year in years:
            all_costs[year] = all_costs[years[0]]

    return all_costs


def update_other_costs(n,costs, years):
    """Update all costs according to the first investment period."""
    for c in n.iterate_components(n.one_port_components|n.branch_components):
        df = c.df
        if (df.empty or (c.name in(["Line", "StorageUnit", "Load"]))): continue
        capital_map =  c.df["capital_cost"].groupby([c.df.carrier, c.df.build_year]).first().sort_index().groupby(level=0).first()
        c.df["capital_cost"] = c.df.carrier.map(capital_map)
        # c.df["capital_cost"] = c.df.carrier.replace(map_dict).map(costs[years[0]]["fixed"]).fillna(c.df.capital_cost)
        if "efficiency" in c.df.columns:
            c.df["efficiency"] = c.df.carrier.replace(map_dict).map(costs[years[0]]["efficiency"]).fillna(c.df.efficiency)
        if "efficiency2" in c.df.columns:
            efficiency_map = c.df["efficiency2"].groupby([c.df.carrier, c.df.build_year]).first().sort_index().groupby(level=0).first()
            c.df["efficiency2"] = c.df.carrier.map(efficiency_map)
            efficiency_map = c.df["efficiency3"].groupby([c.df.carrier, c.df.build_year]).first().sort_index().groupby(level=0).first()
            c.df["efficiency3"] = c.df.carrier.map(efficiency_map)
            efficiency_map = c.df["efficiency4"].groupby([c.df.carrier, c.df.build_year]).first().sort_index().groupby(level=0).first()
            c.df["efficiency4"] = c.df.carrier.map(efficiency_map)
        c.df["lifetime"] = c.df.carrier.replace(map_dict).map(costs[years[0]]["lifetime"]).fillna(c.df.lifetime)
        marginal_map =  c.df["marginal_cost"].groupby([c.df.carrier, c.df.build_year]).first().sort_index().groupby(level=0).first()
        c.df["marginal_cost"] = c.df.carrier.map(marginal_map)

    # electrolysis costs overwrite
    h2_elec_i = n.links[n.links.carrier=="H2 electrolysis"].index
    n.links.loc[h2_elec_i, "capital_cost"] = costs[years[0]].loc["electrolysis", "fixed"]


def prepare_data(gf_default=0.3):
    """Prepare data for multi-decade and technology learning.

    Parameters:
        Returns:
            countries: considered countries
            global_capacity: global installed capacity
            local_capacity: current capacities in countries
            p_nom: used to adjust powerplantmatching data to IRENA data
            p_nom_max_limit: max potential for renewables at one bus
    """
    # considered countries
    countries = n.buses.country.unique()
    n.generators["country"] = n.generators.bus.map(n.buses.country)
    # read in current installed capacities
    global_capacity = pd.read_csv(snakemake.input.global_capacity, index_col=0).iloc[:,0]
    # rename inverter -> charger
    global_capacity.rename(index={"battery inverter": "battery charger"}, inplace=True)
    global_capacity.loc["H2 Fuel Cell"] = global_capacity.loc["H2 fuel cell"]
    local_capacity = pd.read_csv(snakemake.input.local_capacity, index_col=0)

    if all(countries==["EU"]):
        local_capacity["alpha_2"] = "EU"
    else:
        local_capacity = local_capacity[local_capacity.alpha_2.isin(countries)]
    # data to adjust already installed capacity according to IRENA
    local_caps = local_capacity.rename(index={"offwind": "offwind-dc"}).set_index('alpha_2', append=True)
    local_caps = local_caps.groupby(level=[0,1]).sum()

    p_nom = (local_caps.reindex(n.generators.set_index(["carrier", "country"]).index)
            .set_index(n.generators.index)['Electricity Installed Capacity (MW)'])
    p_nom = p_nom[n.generators.build_year<=years[0]]
    p_nom.loc["nuclear {}".format(years[0])] = local_caps.loc["nuclear", 'Electricity Installed Capacity (MW)'][0]
    # from https://beyond-coal.eu/wp-content/uploads/2021/05/2021-04-20_Europe_Beyond_Coal-European_Coal_Database_hcnn.xlsx
    # sheet country (April 2021)
    p_nom.loc["lignite {}".format(years[0])] = 49760
    p_nom.loc["coal {}".format(years[0])] = 76644  # hard coal
    # for GlobalConstraint of the technical limit at each node, get the p_nom_max
    p_nom_max_limit = n.generators.p_nom_max.groupby([n.generators.carrier,
                                                      n.generators.bus,
                                                      n.generators.build_year]).sum()
    p_nom_max_limit = p_nom_max_limit.xs(years[0], level=2)

    # global factor
    global_factor = (local_capacity.groupby(local_capacity.index).sum()
                     .div(global_capacity, axis=0).fillna(gf_default).iloc[:,0])

    return (countries, global_capacity, local_caps, p_nom, p_nom_max_limit,
            global_factor)


def update_network_p_nom(n, p_nom):
    """Prepare network for multi-decade.

    (1) Assign installed capacity align with IRENA data, drops old global
        constraints
    (2) drop old global constraints
    """
    # overwrite already installed renewable capacities
    renewable = ['onwind 2020', 'offwind-dc 2020', 'solar 2020']
    for res in renewable:
        n.generators.loc[renewable, "p_nom"] = p_nom.loc[renewable].astype(float)
        n.generators.loc[renewable, "p_nom_extendable"] = False
        n.generators.loc[renewable, "build_year"] = 2010

        # n.generators.loc[df.index, "p_nom"] = 0.
        # pnl = n.generators_t.p_max_pu[df.index].rename(columns=lambda x: x.replace(" 2020", ""))
        # df.rename(index= lambda x: x.replace(" 2020", ""), inplace=True)

        # df["p_nom"] = p_nom.reindex(df.index).fillna(0).astype(float)

        # n.madd("Generator",
        #        df.index,
        #        bus=df.bus,
        #        carrier=df.carrier,
        #        p_nom_extendable=False,
        #        marginal_cost=df.marginal_cost,
        #        capital_cost=df.capital_cost,
        #        efficiency=df.efficiency,
        #        p_max_pu=pnl)

    # overwrite already installed conventional capacities
    conventional = ["nuclear 2020", "lignite 2020", "coal 2020"]
    efficiencies = n.links.loc[conventional, "efficiency"]
    n.links.loc[conventional, "p_nom"] = p_nom.loc[conventional].astype(float).div(efficiencies, axis=0)
    n.links.loc[conventional, "p_nom_extendable"] = False
    # drop old global constraints
    n.global_constraints.drop(n.global_constraints.index, inplace=True)


def set_scenario_opts(n, opts):
    """Set scenario options."""
    for o in opts:
        # learning
        if "learn" in o:
            techs = list(snakemake.config["learning_rates"].keys())
            if any([tech in o.replace("x", " ") for tech in techs]):
                tech = max([tech for tech in techs if tech in o.replace("x", " ")],
                           key=len)
                learn_diff = float(o[len("learn"+tech):].replace("p",".").replace("m","-"))
                learning_rate = snakemake.config["learning_rates"][tech] + learn_diff
                factor = global_factor.loc[tech]
                if "local" in opts:
                    factor = 1.
                logger.info("technology learning for {} with learning rate {}%".format(tech, learning_rate*100))
                if tech not in n.carriers.index:
                    n.add("Carrier",
                          name=tech)
                n.carriers.loc[tech, "learning_rate"] = learning_rate
                n.carriers.loc[tech, "global_capacity"] = global_capacity.loc[tech]
                n.carriers.loc[tech, "max_capacity"] = 30 * global_capacity.loc[tech]
                n.carriers.loc[tech, "global_factor"] = factor
                for c in nominal_attrs.keys():
                    ext_i = get_extendable_i(n, c)
                    if "carrier" not in n.df(c) or n.df(c).empty: continue
                    learn_assets = n.df(c)[n.df(c)["carrier"]==tech].index
                    learn_assets = ext_i.intersection(n.df(c)[n.df(c)["carrier"]==tech].index)
                    if learn_assets.empty: continue
                    index = n.df(c).loc[learn_assets][n.df(c).loc[learn_assets, "build_year"]<=years[0]].index
                    n.carriers.loc[tech, "initial_cost"] = n.df(c).loc[index, "capital_cost"].mean()
                # TODO
                if tech=="H2 electrolysis":
                    n.carriers.loc["H2 electrolysis", "max_capacity"] = 1.2e6/factor
                if tech=="H2 Fuel Cell":
                    n.carriers.loc["H2 Fuel Cell", "max_capacity"] = 2e4
                if tech=="DAC":
                    n.carriers.loc["DAC", "max_capacity"] = 120e3/factor
                if tech=="solar":
                    n.carriers.loc["solar", "max_capacity"] =  2.2e6/factor
                if tech=="onwind":
                    n.carriers.loc["onwind", "max_capacity"] = 2.3e6/factor

        if "co2seq" in o:
            factor = float(o.replace("co2seq", ""))
            n.stores.loc["co2 stored 2020", "e_nom_max"] *= factor
            logger.info("Total CO2 sequestration potential is set to {}"
                        .format(n.stores.loc["co2 stored 2020", "e_nom_max"]))
    if "local" in opts:
        learn_i = n.carriers[n.carriers.learning_rate!=0].index
        n.carriers.loc[learn_i, "global_factor"] = 1.

    return n


def set_temporal_aggregation(n, opts):
    """Aggregate network temporally."""
    for o in opts:
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
        # typical periods
        m = re.match(r'^\d+p\d\d+h', o, re.IGNORECASE)
        if m is not None:
            opts_t = snakemake.config['temporal_aggregation']
            n_periods = int(o.split("p")[0])
            hours = int(o.split("p")[1].split("h")[0])
            clusterMethod = opts_t["clusterMethod"].replace("-", "_")
            extremePeriodMethod = opts_t["extremePeriodMethod"].replace("-", "_")
            kind = opts_t["kind"]

            logger.info("\n------temporal clustering----------------------------\n"
                        "aggregrate network to {} periods with length {} hours. \n"
                        "Cluster method: {}\n"
                        "extremePeriodMethod: {}\n"
                        "optimisation kind: {}\n"
                        "------------------------------------------------------\n"
                        .format(n_periods, hours, clusterMethod, extremePeriodMethod, kind))

            aggregate_snapshots(n, n_periods=n_periods, hours=hours, clusterMethod=clusterMethod,
                        extremePeriodMethod=extremePeriodMethod)
    return n


def set_multi_index(n, years, social_discountrate):
    """Set snapshots to pd.MultiImdex."""
    snapshots = pd.MultiIndex.from_product([years, n.snapshots])
    n.set_snapshots(snapshots)

    # set investment_weighting
    n.investment_period_weightings.loc[:, "time_weightings"] = n.investment_period_weightings.index.to_series().diff().shift(-1).fillna(10).values
    n.investment_period_weightings.loc[:, "objective_weightings"] = get_investment_weighting(n.investment_period_weightings["time_weightings"], social_discountrate)


def phase_out(c, build_year, lifetime,
              conventionals = ["lignite", "coal", "oil", "nuclear", "CCGT"]):
    """Define phase out time."""
    gens = n.df(c)[ n.df(c).carrier.isin(conventionals)].index
    n.df(c).loc[gens, "build_year"] = build_year
    n.df(c).loc[gens, "lifetime"] = lifetime


def concat_networks(override_component_attrs,
                    with_time=True,
                    snapshots=None, investment_periods=None,
                    ignore_standard_types=False):
        """Concat given pypsa networks and adds build_year.

        Returns
        --------
        network : pypsa.Network

        Parameters
        ----------
        with_time : boolean, default True
            Copy snapshots and time-varying network.component_names_t data too.
        snapshots : list or index slice
            A list of snapshots to copy, must be a subset of
            network.snapshots, defaults to network.snapshots
        ignore_standard_types : boolean, default False
            Ignore the PyPSA standard types.

        Examples
        --------
        >>> network_copy = network.copy()

        """
        for i, network_path in enumerate(snakemake.input.network):
            year = years[i]
            network = pypsa.Network(network_path,
                                    override_component_attrs=override_component_attrs)
            # initialise final network n
            if i ==0:
                override_components, override_component_attrs = network._retrieve_overridden_components()

                n = network.__class__(ignore_standard_types=ignore_standard_types,
                                     override_components=override_components,
                                     override_component_attrs=override_component_attrs)


            for component in network.iterate_components(["Bus", "Carrier"]):
                # add missing assets
                df_year = component.df
                df_final = getattr(n, component.list_name)
                missing_i = df_year.index.difference(df_final.index)
                missing = df_year.loc[missing_i]
                if missing.empty: continue
                import_components_from_dataframe(n, missing, component.name)

            for component in network.iterate_components(['Generator', 'Link', 'Store', 'Load']):

                df_year = component.df
                df_final = getattr(n, component.list_name)
                df_year.rename(index= lambda x: x.replace("-2020","") + " " + str(year),
                               inplace=True)
                df_year["build_year"] = year

                import_components_from_dataframe(n, df_year, component.name)

            if with_time:
                if snapshots is None:
                    snapshots = network.snapshots
                n.set_snapshots(snapshots)
                investment_periods = network.investment_period_weightings.index
                for component in network.iterate_components():
                    pnl = getattr(n, component.list_name+"_t")
                    for k in iterkeys(component.pnl):
                        if component.name in ['Generator', 'Link', 'Store', 'Load']:
                            pnl_year = (component.pnl[k].loc[snapshots].copy()
                                        .rename(columns = lambda x: x.replace("-2020","")  + " " + str(year)))
                            pnl[k] = pd.concat([pnl[k], pnl_year], axis=1)
                        elif component.name in ["Bus", "Carrier"]:
                            df_year = component.df
                            df_final = getattr(n, component.list_name)
                            missing_i = df_year.index.difference(df_final.index)
                            missing = df_year.loc[missing_i]
                            if missing.empty: continue
                            cols = missing.index.intersection(component.pnl[k].columns)
                            if cols.empty: continue
                            pnl_year = (component.pnl[k].loc[snapshots].copy()
                                        .rename(columns = lambda x: x.replace("-2020","")  + " " + str(year)))
                            pnl[k] = pd.concat([pnl[k], pnl_year[cols]], axis=1)
                        else:
                            pnl[k] = component.pnl[k]

                n.snapshot_weightings = network.snapshot_weightings.loc[snapshots].copy()
                n.investment_period_weightings = network.investment_period_weightings.loc[investment_periods].copy()
            else:
                investment_periods = network.investment_period_weightings.index
                n.snapshot_weightings = network.snapshot_weightings.copy()
                n.investment_period_weightings = network.investment_period_weightings.loc[investment_periods].copy()


        # TODO some renaming
        battery_i = n.links[n.links.carrier=="battery charger"].index
        n.links.loc[battery_i, "efficiency"] = n.links.loc[battery_i, "efficiency"]**0.5
        n.links.rename({"H2 Electrolysis": "H2 electrolysis"}, inplace=True)
        n.links.carrier.replace({"H2 Electrolysis": "H2 electrolysis"}, inplace=True)
        n.links.carrier.replace({"home battery charger": "battery charger"}, inplace=True)

        return n


def set_fixed_assets(c, links_fixed):
    fix = n.df(c)[n.df(c).carrier.isin(links_fixed)]
    keep = fix[fix.build_year<=years[0]]
    caps_i = keep[keep.build_year<years[0]]
    keep.drop(keep[keep.carrier.isin(caps_i.index)].index, inplace=True)
    keep = pd.concat([caps_i, keep])
    to_drop = fix.index.difference(keep.index)
    for asset in to_drop:
        n.remove(c, asset)
    n.df(c).loc[keep.index, "lifetime"] = np.inf

def add_battery_constraints(n):

    chargers = n.links.index[n.links.carrier.str.contains("battery charger") & n.links.p_nom_extendable]
    dischargers = chargers.str.replace("charger","discharger")

    link_p_nom = get_var(n, "Link", "p_nom")

    lhs = linexpr((1,link_p_nom[chargers]),
                  (-n.links.loc[dischargers, "efficiency"].values,
                   link_p_nom[dischargers].values))

    define_constraints(n, lhs, "=", 0, 'Link', 'charger_ratio')


def get_nodal_balance(carrier="gas"):

    bus_map = (n.buses.carrier == carrier)
    bus_map.at[""] = False
    supply_energy = pd.Series(dtype="float64")

    for c in n.iterate_components(n.one_port_components):

        items = c.df.index[c.df.bus.map(bus_map).fillna(False)]

        if len(items) == 0:
            continue

        s = round(c.pnl.p.multiply(n.snapshot_weightings.generator_weightings,axis=0).sum().multiply(c.df['sign']).loc[items]
             .groupby([c.df.bus, c.df.carrier]).sum())
        s = pd.concat([s], keys=[c.list_name])
        s = pd.concat([s], keys=[carrier])

        supply_energy = supply_energy.reindex(s.index.union(supply_energy.index))
        supply_energy.loc[s.index] = s


    for c in n.iterate_components(n.branch_components):

        for end in [col[3:] for col in c.df.columns if col[:3] == "bus"]:

            items = c.df.index[c.df["bus" + str(end)].map(bus_map,na_action=False)]

            if len(items) == 0:
                continue

            s = ((-1)*c.pnl["p"+end][items].multiply(n.snapshot_weightings.generator_weightings,axis=0).sum()
                .groupby([c.df.loc[items,'bus{}'.format(end)], c.df.loc[items,'carrier']]).sum())
            s.index = s.index
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[carrier])

            supply_energy = supply_energy.reindex(s.index.union(supply_energy.index))

            supply_energy.loc[s.index] = s

    supply_energy = supply_energy.rename(index=lambda x: rename_techs(x), level=3)
    return supply_energy


def add_carbon_neutral_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "Co2Neutral"')
    for name, glc in glcs.iterrows():
        rhs = glc.constant
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f'{carattr} != 0')[carattr]

        if emissions.empty: continue

        # stores
        n.stores['carrier'] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query('carrier in @emissions.index and not e_cyclic')
        time_valid = int(glc.loc["investment_period"])
        if not stores.empty:
            final_e = get_var(n, 'Store', 'e').groupby(level=0).last()[stores.index]
            lhs = linexpr((-1, final_e.shift().loc[time_valid]),
                          (1, final_e.loc[time_valid]))
            define_constraints(n, lhs, "==", rhs, 'GlobalConstraint', 'Co2Neutral')


def add_local_res_constraint(n,snapshots):

    c, attr = 'Generator', 'p_nom'
    res = ['offwind-ac', 'offwind-dc', 'onwind', 'solar', 'solar rooftop']
    ext_i = n.df(c)[(n.df(c)["carrier"].isin(res))
                    & (n.df(c)["country"] !="EU")
                    & (n.df(c)["p_nom_extendable"])].index
    time_valid = snapshots.levels[0]

    active_i = pd.concat([get_active_assets(n,c,inv_p,snapshots).rename(inv_p)
                          for inv_p in time_valid], axis=1).astype(int)

    ext_and_active = active_i.T[active_i.index.intersection(ext_i)]

    if ext_and_active.empty: return

    cap_vars = get_var(n, c, attr)[ext_and_active.columns]

    lhs = (linexpr((ext_and_active, cap_vars)).T
           .groupby([n.df(c).carrier, n.df(c).country]).sum().T)

    p_nom_max_w = n.df(c).p_nom_max.div(n.df(c).weight).loc[ext_and_active.columns]
    p_nom_max_t = expand_series(p_nom_max_w, time_valid).T

    rhs = (p_nom_max_t.mul(ext_and_active)
           .groupby([n.df(c).carrier, n.df(c).country], axis=1)
           .max())

    define_constraints(n, lhs, "<=", rhs, 'GlobalConstraint', 'res_limit')

#%%
# for testing
if 'snakemake' not in globals():
    os.chdir("/home/ws/bw0928/Dokumente/learning_curve/scripts")
    from _helpers import mock_snakemake
    snakemake = mock_snakemake('solve_sec_network_years',
                               sector_opts= '4p24h-learnH2xelectrolysisp0-co2seq1',
                               clusters='37')

# parameters
years = snakemake.config["scenario"]["investment_periods"]
social_discountrate = snakemake.config["costs"]["social_discountrate"]
# scenario options
opts = snakemake.wildcards.sector_opts.split('-')

n = concat_networks(override_component_attrs)

# costs
update = snakemake.config["costs"]["update_costs"]
costs = prepare_costs_all_years(years, update)
# prepare data
representative_ct = ['DE', 'DK', 'GB', 'IT','ES', 'PT', 'PL']
# representative_ct = ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
#        'FR', 'GB', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'ME', 'MK',
#        'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SE', 'SI', 'SK']
countries, global_capacity, local_capacity, p_nom, p_nom_max_limit, global_factor = prepare_data()

# representative renewable for different region
p_max_pu = pd.read_csv(snakemake.input.p_max_pu,
                       index_col=0, header=[0,1], parse_dates=True)
static_df = pd.read_csv(snakemake.input.generators_costs,
                        index_col=[0,1])
for carrier in p_max_pu.columns.levels[0]:
    if carrier=="ror": continue
    df = n.generators[n.generators.carrier==carrier]
    pnl = p_max_pu[carrier].reindex(columns=representative_ct).dropna(axis=1)
    df_cts = pd.concat([df.rename(index=lambda x: ct + " " + x) for ct in pnl.columns])
    df_cts["country"] = df_cts.index.str[:2]
    pnl_cts = pd.concat([pnl.rename(columns=lambda x: x + " " + name)
                     for name in df.index], axis=1).reindex(columns=df_cts.index)
    attributes = ["capital_cost", "p_nom_max"]
    reindex_df = df_cts.set_index(["carrier", "country"]).rename({"offwind":"offwind-dc"},level=0)
    for attr in attributes:
        replace = static_df.reindex(reindex_df.index)[attr]
        replace.index = df_cts.index
        df_cts[attr] = replace
    # share of total European technical potential
    if p_nom_max_limit.loc[carrier][0]!= np.inf:
        df_cts["weight"] = df_cts[df_cts.build_year==years[0]].p_nom_max.sum() / p_nom_max_limit.loc[carrier][0]


    n.madd("Generator",
           df_cts.index,
           bus=df_cts.bus,
           carrier=carrier,
           p_nom_extendable=df_cts.p_nom_extendable,
           p_nom_max=df_cts.p_nom_max,
           weight=df_cts.weight,
           build_year=df_cts.build_year,
           lifetime=df_cts.lifetime,
           country=df_cts.country,
           marginal_cost=df_cts.marginal_cost,
           capital_cost=df_cts.capital_cost,
           efficiency=df_cts.efficiency,
           p_max_pu=pnl_cts)

# update already installed capacities
update_network_p_nom(n, p_nom)
# update costs
update_other_costs(n, costs, years)

# set scenario options
n = set_scenario_opts(n, opts)
# aggregate network temporal
n = set_temporal_aggregation(n, opts)

# set snapshots MultiIndex and investment weightings
set_multi_index(n, years, social_discountrate)

# take care of loads --------------------------------------------------------
loads = (n.loads_t.p_set.groupby([n.loads.carrier, n.loads.build_year], axis=1)
         .sum().loc[years[0]].stack().swaplevel().groupby(level=[0,1]).first())
loads.fillna(0, inplace=True)

keep  = (n.loads.loc[n.loads_t.p_set.columns]
         .groupby([n.loads.carrier, n.loads.build_year]).sum()
         .groupby(level=0).first())
keep.index = keep.index + " " + keep.build_year.astype(str)
drop = n.loads_t.p_set.columns.difference(keep.index)
n.loads.drop(drop, inplace=True)
n.loads_t.p_set = loads
n.loads.rename(index=lambda x: x[:-5] if x in keep.index else x,
               inplace=True)
for carrier in n.loads[n.loads.p_set!=0].carrier.unique():
    to_change = n.loads[n.loads.carrier==carrier]
    # load stays constant during different investment periods
    if all(to_change.p_set==to_change.p_set[0]):
        n.loads.drop(to_change.index[1:], inplace=True)
    # time variant load
    else:
        p_set = (to_change.p_set.rename(index=to_change.build_year)
                 .reindex(n.snapshots, level=0))
        n.loads.drop(to_change.index[1:], inplace=True)
        n.loads.loc[to_change.index[0], "p_set"] = 0
        n.loads_t.p_set[to_change.index[0]] = p_set

# ### Play around with assumptions: ------------------------------------
# 1) conventional phase out
# phase_out(2013, 20)

# set conventional assumptions
conventionals = ['coal', 'gas', 'lignite', 'oil', 'ror', 'uranium']
set_fixed_assets("Generator", conventionals)

links_fixed = ['CCGT',
       'biogas to gas',
       'co2 vent', 'coal', 'electricity distribution grid',
       'gas for industry', 'gas for industry CC',
       'lignite',
       'nuclear', 'oil', 'process emissions', 'process emissions CC',
       'residential rural water tanks charger',
       'residential rural water tanks discharger',
       'residential urban decentral water tanks charger',
       'residential urban decentral water tanks discharger',
       'services rural water tanks charger',
       'services rural water tanks discharger',
       'services urban decentral water tanks charger',
       'services urban decentral water tanks discharger',
       'solid biomass for industry', 'solid biomass for industry CC',
       'urban central water tanks charger',
       'urban central water tanks discharger']

set_fixed_assets("Link", links_fixed)


stores_fixed = ['co2', 'co2 stored', 'coal', 'gas', 'lignite', 'oil', 'uranium',
                'H2 Store']
set_fixed_assets("Store", stores_fixed)


phase_out_carrier = ['CCGT', 'coal', 'lignite', 'nuclear']
phase_out("Link", 2010, 20, phase_out_carrier)

n.links.lifetime.fillna(10., inplace=True)

# co2 assumptions
n.stores.loc["co2 2020", "e_period"] = False
n.stores.loc["co2 stored 2020", "e_period"] = False
#
# increase biomass potentials
for carrier in ["biogas", "solid biomass"]:
    store_i = n.stores[n.stores.carrier==carrier].index
    n.stores.loc[store_i, ["e_nom", "e_initial"]] = n.stores.loc[store_i, ["e_nom", "e_initial"]].mul(n.investment_period_weightings["time_weightings"].rename(lambda x: carrier+ " " +str(x)), axis=0)
    n.stores.loc[store_i, "lifetime"] = 10.

# (a) add CO2 Budget constraint ------------------------------------
budget = snakemake.config["co2_budget"]["2p0"] * 1e9  # budget for + 1.5 Celsius for Europe


n.add("GlobalConstraint",
      "Budget",
      type="primary_energy",
      carrier_attribute="co2_emissions",
      sense="<=",
      constant=budget)

n.add("GlobalConstraint",
      "Co2neutral",
      type="Co2Neutral",
      carrier_attribute="co2_emissions",
      investment_period=n.snapshots.levels[0][-1],
      sense="<=",
      constant=0)



# global p_nom_max for each carrier + investment_period at each node
p_nom_max_inv_p = pd.DataFrame(np.repeat([p_nom_max_limit.values],
                                         len(n.snapshots.levels[0]), axis=0),
                               index=n.snapshots.levels[0], columns=p_nom_max_limit.index)

renewables = ['offwind-ac', 'offwind-dc', 'onwind','solar',
              'solar rooftop']
if snakemake.config["tech_limit"]:
    logger.info("set technical potential at each node")
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

limit_res = ['offwind-ac', 'offwind-dc', 'onwind','solar']
# TODO limit max growth per year
# solar max grow so far 28 GW in Europe https://www.iea.org/reports/renewables-2020/solar-pv
n.carriers.loc["solar", "max_growth"] = 40 * 1e3
# onshore max grow so far 16 GW in Europe https://www.iea.org/reports/renewables-2020/wind
n.carriers.loc["onwind", "max_growth"] = 30 * 1e3
# offshore max grow so far 3.5 GW in Europe https://windeurope.org/about-wind/statistics/offshore/european-offshore-wind-industry-key-trends-statistics-2019/
n.carriers.loc[['offwind-ac', 'offwind-dc'], "max_growth"] = 6 * 1e3

# if "H2 electrolysis" not in n.carriers.index:
#     n.add("Carrier",
#           name="H2 electrolysis" )
# n.carriers.loc["H2 electrolysis", "max_growth"] = 260 * 1e3
#%%---------------------------------------------------------------------------
if any(n.carriers.learning_rate!=0):
    def extra_functionality(n, snapshots):
        add_battery_constraints(n)
        add_learning(n, snapshots)
        add_carbon_neutral_constraint(n, snapshots)
        add_local_res_constraint(n,snapshots)

    skip_objective = True
else:
    def extra_functionality(n, snapshots):
        add_battery_constraints(n)
        add_carbon_neutral_constraint(n, snapshots)
        add_local_res_constraint(n,snapshots)

    skip_objective = False

# check for typcial periods
if hasattr(n, "cluster"):
    typical_period=True
    # TODO
    typical_period=False
else:
    typical_period=False

config = snakemake.config['solving']
solve_opts = config['options']
solver_options = config['solver'].copy()
solver_log = snakemake.log.solver
solver_options["threads"] = snakemake.threads

# MIPFocus = 3, MIPGap, MIRCuts=2 (agressive)
dac_i =  n.links[n.links.carrier=="DAC"].index
n.links.loc[dac_i, "bus3"] = 'services urban decentral heat'

logging.basicConfig(filename=snakemake.log.python,
                    level=snakemake.config['logging']['level'])

with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

    n = prepare_network(n, solve_opts)
    #%%
    n.lopf(pyomo=False, solver_name="gurobi", skip_objective=skip_objective,
           multi_investment_periods=True, solver_options=solver_options,
           solver_logfile=solver_log,
           extra_functionality=extra_functionality, keep_shadowprices=False,
           typical_period=typical_period)

    n.export_to_netcdf(snakemake.output[0])

    # for key in n.sols["Carrier"]["pnl"]:
    #     data = round(n.sols["Carrier"]["pnl"][key].groupby(level=0).mean())
    #     data.to_csv("/home/ws/bw0928/Dokumente/learning_curve/results/test/csvs/{}.csv".format(key))


logger.info("Maximum memory usage: {}".format(mem.mem_usage))
