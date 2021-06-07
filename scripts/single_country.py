#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
the following script should convert a whole European pypsa-eur-sec
network to a single country network

Created on Sun Jun  6 08:41:39 2021
@author: bw0928
"""

import pypsa
import numpy as np
import pandas as pd

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

override_component_attrs["Link"].loc["build_year"] = ["integer","year",np.nan,"build year","Input (optional)"]
override_component_attrs["Link"].loc["lifetime"] = ["float","years",np.nan,"lifetime","Input (optional)"]
override_component_attrs["Generator"].loc["build_year"] = ["integer","year",np.nan,"build year","Input (optional)"]
override_component_attrs["Generator"].loc["lifetime"] = ["float","years",np.nan,"lifetime","Input (optional)"]
override_component_attrs["Store"].loc["build_year"] = ["integer","year",np.nan,"build year","Input (optional)"]
override_component_attrs["Store"].loc["lifetime"] = ["float","years",np.nan,"lifetime","Input (optional)"]


#%%
path_europe_network = "/home/ws/bw0928/Dokumente/pypsa-eur/networks/elec_s_37_ec_lv1.0_.nc"
path_out = "/home/ws/bw0928/Dokumente/learning_curve/results/prenetworks/"
country = ["DE"]

n = pypsa.Network(path_europe_network,
                  override_component_attrs=override_component_attrs)

# drop other buses
n.buses = n.buses[n.buses.country.isin(country + ["EU"])]

# drop other buses in network components
for component in (n.one_port_components | {"Load"} | n.branch_components):
    df = n.df(component)

    if component in n.branch_components:
        to_drop = df[~(df.bus0.apply(lambda x: any(c in x for c in n.buses.index)) &
             df.bus1.apply(lambda x: any(c in x for c in n.buses.index)))].index
    else:
        to_drop = df[~df.bus.isin(n.buses.index)].index
        df = df[df.bus.isin(n.buses.index)]

    for asset in to_drop:
        n.remove(component, asset)

n.export_to_netcdf(path_out+"{}.nc".format(country))

