#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 07:33:10 2021

@author: bw0928
"""
import os
import pandas as pd
import numpy as np

import logging

import pypsa
from pypsa.io import import_components_from_dataframe

from distutils.version import LooseVersion
pd_version = LooseVersion(pd.__version__)
agg_group_kwargs = dict(numeric_only=False) if pd_version >= "1.3" else {}

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
if 'snakemake' not in globals():
    os.chdir("/home/ws/bw0928/Dokumente/learning_curve/scripts")
    from _helpers import mock_snakemake
    # snakemake = mock_snakemake('solve_network', lv='1.0', clusters='37',
    #                            sector_opts='Co2L-2p24h-learnsolarp0')
    snakemake = mock_snakemake('prepare_sec_network',
                               sector_opts= 'Co2L-2p24h-learnsolarp0-learnbatteryp0-learnonwindp0-learnH2xelectrolysisp0',
                               clusters='37')

# import pypsa network
n = pypsa.Network(snakemake.input.network,
                  override_component_attrs=override_component_attrs)
#%%
aggregate_dict = {"p_nom": "sum",
                  "p_nom_max": "sum",
                  "p_nom_min": "sum",
                  "p_nom_max": "sum",
                  "p_set": "sum",
                  "e_initial": "sum",
                  "e_nom": "sum",
                  "e_nom_max": "sum",
                  "e_nom_min": "sum",
                  "state_of_charge_initial": "sum",
                  "state_of_charge_set":"sum",
                  "inflow": "sum",
                  "p_max_pu": "mean"}

# group all components to one european bus
m = pypsa.Network(override_component_attrs=override_component_attrs)

# set snapshots
m.set_snapshots(n.snapshots)
m.snapshot_weightings = n.snapshot_weightings.copy()

 #catch all remaining attributes of network
for attr in ["name", "srid"]:
    setattr(m,attr,getattr(n,attr))

other_comps = sorted(n.all_components - {"Bus","Carrier"} - {"Line"})
# overwrite static component attributes
for component in n.iterate_components(["Bus", "Carrier"] + other_comps):
    df = component.df
    default = n.components[component.name]["attrs"]["default"]
    for col in df.columns.intersection(default.index):
        df[col].fillna(default.loc[col], inplace=True)
    if hasattr(df, "build_year"):
        df["build_year"].fillna(0., inplace=True)
        df["lifetime"].fillna(np.inf, inplace=True)
    if hasattr(df, "carrier"):
        keys = df.columns.intersection(aggregate_dict.keys())
        agg = dict(zip(df.columns.difference(keys), ["first"]*len(df.columns.difference(keys))))
        for key in keys:
            agg[key] = aggregate_dict[key]
        df = df.groupby("carrier").agg(agg, **agg_group_kwargs)
        # rename location
        df["country"] = "EU"
        df["location"] = "EU"
        df["carrier"] = df.index
        # rename buses
        df.loc[:,df.columns.str.contains("bus")] = (df.loc[:,df.columns.str.contains("bus")]
                                                   .apply(lambda x: x.map(n.buses.carrier)))
    #drop the standard types to avoid them being read in twice
    if component.name in n.standard_type_components:
        df = component.df.drop(m.components[component.name]["standard_types"].index)

    import_components_from_dataframe(m, df, component.name)

# time varying data
for component in n.iterate_components():
    pnl = getattr(m, component.list_name+"_t")
    df = component.df
    if not hasattr(df, "carrier"): continue
    keys = pd.Index(component.pnl.keys()).intersection(aggregate_dict.keys())
    agg = dict(zip(pd.Index(component.pnl.keys()).difference(aggregate_dict.keys()),
                   ["first"]*len(pd.Index(component.pnl.keys()).difference(aggregate_dict.keys()))))
    for key in keys:
        agg[key] = aggregate_dict[key]

    for k in component.pnl.keys():
        pnl[k] = component.pnl[k].groupby(df.carrier,axis=1).agg(agg[k], **agg_group_kwargs)
        pnl[k].fillna(n.components[component.name]["attrs"].loc[k, "default"], inplace=True)

to_drop = ["H2 pipeline", "H2 pipeline retrofitted", "Gas pipeline", "DC"]
m.links.drop(to_drop, inplace=True, errors="ignore")

# drop old global constraints
m.global_constraints.drop(m.global_constraints.index, inplace=True)

# adjust p nom max
p_nom_max = n.generators[n.generators.p_nom_extendable].groupby(n.generators.carrier).sum(**agg_group_kwargs).p_nom_max
m.generators.loc[p_nom_max[p_nom_max!=np.inf].index, "p_nom_max"] = p_nom_max[p_nom_max!=np.inf]
#%%
m.export_to_netcdf(snakemake.output[0])
