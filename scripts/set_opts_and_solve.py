#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 16:52:45 2021

@author: bw0928
"""
import pypsa_learning as pypsa
import pandas as pd
import numpy as np
import logging
from six import iteritems, string_types
import re

from pypsa_learning.io import import_components_from_dataframe
from pypsa_learning.descriptors import (
    expand_series,
    nominal_attrs,
    get_extendable_i,
    get_active_assets,
)
from pypsa_learning.linopt import (
    get_var,
    linexpr,
    define_constraints,
    write_constraint,
    set_conref,
)
from pypsa_learning.temporal_clustering import aggregate_snapshots
from pypsa_learning.learning import (
    add_learning,
    experience_curve,
    get_linear_interpolation_points,
    get_slope,
    cumulative_cost_curve
)

from distutils.version import LooseVersion

from prepare_perfect_foresight import prepare_costs_all_years, pypsa_to_techbase

pd_version = LooseVersion(pd.__version__)
agg_group_kwargs = dict(numeric_only=False) if pd_version >= "1.3" else {}

from vresutils.benchmark import memory_logger

logger = logging.getLogger(__name__)

# TODO move override component to helpers
override_component_attrs = pypsa.descriptors.Dict(
    {k: v.copy() for k, v in pypsa.components.component_attrs.items()}
)
override_component_attrs["Link"].loc["bus2"] = [
    "string",
    np.nan,
    np.nan,
    "2nd bus",
    "Input (optional)",
]
override_component_attrs["Link"].loc["bus3"] = [
    "string",
    np.nan,
    np.nan,
    "3rd bus",
    "Input (optional)",
]
override_component_attrs["Link"].loc["bus4"] = [
    "string",
    np.nan,
    np.nan,
    "4th bus",
    "Input (optional)",
]
override_component_attrs["Link"].loc["efficiency2"] = [
    "static or series",
    "per unit",
    1.0,
    "2nd bus efficiency",
    "Input (optional)",
]
override_component_attrs["Link"].loc["efficiency3"] = [
    "static or series",
    "per unit",
    1.0,
    "3rd bus efficiency",
    "Input (optional)",
]
override_component_attrs["Link"].loc["efficiency4"] = [
    "static or series",
    "per unit",
    1.0,
    "4th bus efficiency",
    "Input (optional)",
]
override_component_attrs["Link"].loc["p2"] = [
    "series",
    "MW",
    0.0,
    "2nd bus output",
    "Output",
]
override_component_attrs["Link"].loc["p3"] = [
    "series",
    "MW",
    0.0,
    "3rd bus output",
    "Output",
]
override_component_attrs["Link"].loc["p4"] = [
    "series",
    "MW",
    0.0,
    "4th bus output",
    "Output",
]

import os

os.environ["NUMEXPR_MAX_THREADS"] = str(snakemake.threads)

# FUNCTIONS ---------------------------------------------------------------
aggregate_dict = {
    "p_nom": "sum",
    "s_nom": "sum",
    "v_nom": "max",
    "v_mag_pu_max": "min",
    "v_mag_pu_min": "max",
    "p_nom_max": "sum",
    "s_nom_max": "sum",
    "p_nom_min": "sum",
    "s_nom_min": "sum",
    'v_ang_min': "max",
    "v_ang_max":"min",
    "terrain_factor":"mean",
    "num_parallel": "sum",
    "p_set": "sum",
    "e_initial": "sum",
    "e_nom": "sum",
    "e_nom_max": "sum",
    "e_nom_min": "sum",
    "state_of_charge_initial": "sum",
    "state_of_charge_set": "sum",
    "inflow": "sum",
    # TODO
    "p_max_pu": "first",
    "x": "mean",
    "y": "mean"
}


def heat_must_run(n):
    """
    Force heating technologies to follow heat demand profile.
    """
    logger.info("Add must-run condition for heating technologies.\n")
    cols = n.loads_t.p_set.columns[n.loads_t.p_set.columns.str.contains("heat")
                                   & ~n.loads_t.p_set.columns.str.contains("industry")]
    profile = n.loads_t.p_set[cols] / n.loads_t.p_set[cols].groupby(level=0).max()
    profile.rename(columns=n.loads.bus.to_dict(), inplace=True)

    profile1 = profile.reindex(columns=n.links.bus1)
    profile1.columns = n.links.index
    profile1.drop(n.links[~n.links.p_nom_extendable].index, axis=1, inplace=True)
    profile1.dropna(axis=1, inplace=True)
    to_drop = profile1.columns[profile1.columns.str.contains("water tank")]
    profile1.drop(to_drop, inplace=True, axis=1)

    # n.links_t.p_max_pu = pd.concat([n.links_t.p_max_pu, profile1], axis=1).groupby(level=0, axis=1).min()
    n.links_t.p_min_pu[profile1.columns] = profile1
    # n.links_t.p_max_pu[profile1.columns] = profile1

    profile2 = profile.reindex(columns=n.links.bus2)
    profile2.columns = n.links.index
    profile2.drop(n.links[~n.links.p_nom_extendable].index, axis=1, inplace=True)
    profile2.dropna(axis=1, inplace=True)
    to_drop = profile2.columns[profile2.columns.str.contains("H2 Fuel Cell")]
    profile2.drop(to_drop, inplace=True, axis=1)

    # n.links_t.p_max_pu = pd.concat([n.links_t.p_max_pu, profile2], axis=1).groupby(level=0, axis=1).min()
    n.links_t.p_min_pu[profile2.columns] = profile2
    # n.links_t.p_max_pu[profile2.columns] = profile2

    profile3 = profile.reindex(columns=n.generators.bus)
    profile3.columns = n.generators.index
    profile3.drop(n.generators[~n.generators.p_nom_extendable].index, axis=1, inplace=True)
    profile3.dropna(axis=1, inplace=True)
    # n.generators_t.p_max_pu = pd.concat([n.generators_t.p_max_pu, profile3], axis=1).groupby(level=0, axis=1).min()
    profile3 = pd.concat([n.generators_t.p_max_pu.reindex(columns=profile3.columns),
                          profile3], axis=1).groupby(level=0, axis=1).min()
    n.generators_t.p_min_pu[profile3.columns] = profile3
    n.generators_t.p_max_pu[profile3.columns] = profile3

    return n


def cluster_heat_buses(n):
    """Cluster residential and service heat buses to one representative bus.

    This is done to save memory and speed up optimisation
    """
    logger.info("Cluster residential and service heat buses.")
    assign_location(n)

    components = ["Bus", "Carrier", "Generator", "Link", "Load", "Store"]
    components_t = {"Bus": "buses_t",
                    "Carrier": "carriers_t",
                    "Generator": "generators_t",
                    "Link": "links_t",
                    "Load": "loads_t",
                    "Store":"stores_t"}

    for c in components:
        df = getattr(n, components_t[c][:-2])
        cols = df.columns[df.columns.str.contains("bus") | (df.columns=="carrier")]
        links_i = (df[cols].apply(lambda x: x.str.contains("residential")
                                  | x.str.contains("services"),
                                       axis=1)).any(axis=1)
        if "carrier" in df.columns:
            logger.info("cluster techs: \n {}\n".format(df.loc[links_i, "carrier"].unique()))
        df[cols] = (df[cols]
                         .apply(lambda x: x.str.replace("residential ","")
                                .str.replace("services ", ""), axis=1))
        df = df.rename(index=lambda x: x.replace("residential ","")
                       .replace("services ", ""))
        keys = df.columns.intersection(aggregate_dict.keys())
        agg = dict(
            zip(
                df.columns.difference(keys),
                ["first"] * len(df.columns.difference(keys)),
            )
        )
        for key in keys:
            agg[key] = aggregate_dict[key]

        df = df.groupby(level=0).agg(agg, **agg_group_kwargs)


        # time varying data

        pnl = getattr(n, components_t[c])
        keys = pd.Index(pnl.keys()).intersection(aggregate_dict.keys())
        agg = dict(
            zip(
                pd.Index(pnl.keys()).difference(aggregate_dict.keys()),
                ["first"]
                * len(pd.Index(pnl.keys()).difference(aggregate_dict.keys())),
            )
        )
        for key in keys:
            agg[key] = aggregate_dict[key]

        for k in pnl.keys():
            pnl[k].rename(columns=lambda x: x.replace("residential ","")
                           .replace("services ", ""), inplace=True)
            pnl[k] = (
                 pnl[k]
                .groupby(level=0, axis=1)
                .agg(agg[k], **agg_group_kwargs)
            )

        to_drop = n.df(c).index.difference(df.index)
        n.mremove(c, to_drop)
        to_add = df.index.difference(n.df(c).index)
        import_components_from_dataframe(n, df.loc[to_add], c)

    return n

def cluster_to_regions(n):
    """Cluster European countries to representative regions."""
    assign_location(n)
    cluster_regions = {
        "North": ["NO", "SE", "FI", "EE", "LV", "LT"],
        "East": ["GR", "AL", "MK", "ME", "RS", "BG", "RO",
                 "HR", "HU", "SK", "PL", "CZ", "BA", "SI"],
        "Central": ["DE", "AT", "CH", "LU", "DK", "NL",
                    "BE", "LT"],
        "South": ["FR", "ES", "IT", "PT"],
        "West": ["GB", "IE"]
        }
    ct_to_region = {}
    for region, country in cluster_regions.items():
        countries = cluster_regions[region]
        for ct in countries:
            ct_to_region[ct] = region

    # assign to region -------------------------------------------------------
    n.buses["region"] = n.buses.country.map(ct_to_region)
    n.buses.region.fillna("EU", inplace=True)
    for c in n.one_port_components:
        n.df(c)["region"] = n.df(c).bus.map(n.buses.region)
    for c in n.branch_components:
        n.df(c)["region"] = n.df(c).bus0.map(n.buses.region)
        n.df(c)["region1"] = n.df(c).bus1.map(n.buses.region)
        if hasattr(n.df(c), "bus3"):
            n.df(c)["region3"] = n.df(c).bus3.map(n.buses.region).fillna("")
        eu_i = n.df(c)["country"] == "EU"
        n.df(c).loc[eu_i, "region"] = n.df(c).loc[eu_i, "bus1"].map(n.buses.region)

    # -------------------------------------------------------------------------
    # clustered network m
    m = pypsa.Network(override_component_attrs=override_component_attrs)
    # set snapshots
    m.set_snapshots(n.snapshots)
    m.snapshot_weightings = n.snapshot_weightings.copy()
    m.investment_period_weightings = n.investment_period_weightings.copy()

    # catch all remaining attributes of network
    for attr in ["name", "srid"]:
        setattr(m, attr, getattr(n, attr))

    other_comps = sorted(n.all_components - {"Bus", "Carrier"} - {"Line"})
    # overwrite static component attributes
    for component in n.iterate_components(["Bus", "Carrier"] + other_comps):
        df = component.df
        default = n.components[component.name]["attrs"]["default"]
        for col in df.columns.intersection(default.index):
            df[col].fillna(default.loc[col], inplace=True)
        if hasattr(df, "carrier"):
            keys = df.columns.intersection(aggregate_dict.keys())
            agg = dict(
                zip(
                    df.columns.difference(keys),
                    ["first"] * len(df.columns.difference(keys)),
                )
            )
            for key in keys:
                agg[key] = aggregate_dict[key]
            if hasattr(df, "region1"):
                if hasattr(df, "region3"):
                    df_2 = df[df.carrier.isin(["DAC", "Fischer-Tropsch"])].groupby(["region3", "carrier", "build_year"]).agg(agg, **agg_group_kwargs)
                else:
                    df_2 = pd.DataFrame()
                df = df.groupby(["region1", "carrier", "build_year"]).agg(agg, **agg_group_kwargs)
                df = pd.concat([df.drop(["DAC", "Fischer-Tropsch"], level=1), df_2])
                df.index = pd.Index([f"{region} {i}-{int(j)}" for region, i, j in df.index])
            elif hasattr(df, "build_year"):
                df = df.groupby(["region", "carrier", "build_year"]).agg(agg, **agg_group_kwargs)
                df.index = pd.Index([f"{region} {i}-{int(j)}" for region, i, j in df.index])
            else:
                df = df.groupby(["region", "carrier"]).agg(agg, **agg_group_kwargs)
                df.index = pd.Index([f"{region} {i}" for region, i in df.index])
            # rename buses
            df.loc[:, df.columns.str.contains("bus")] = df.loc[
                :, df.columns.str.contains("bus")
            ].apply(lambda x: x.map(n.buses.region) + " " + x.map(n.buses.carrier))
        # drop the standard types to avoid them being read in twice
        if component.name in n.standard_type_components:
            df = component.df.drop(m.components[component.name]["standard_types"].index)

        import_components_from_dataframe(m, df, component.name)

    # time varying data
    for component in n.iterate_components():
        pnl = getattr(m, component.list_name + "_t")
        df = component.df
        if not hasattr(df, "region"):
            continue
        keys = pd.Index(component.pnl.keys()).intersection(aggregate_dict.keys())
        agg = dict(
            zip(
                pd.Index(component.pnl.keys()).difference(aggregate_dict.keys()),
                ["first"]
                * len(pd.Index(component.pnl.keys()).difference(aggregate_dict.keys())),
            )
        )
        for key in keys:
            agg[key] = aggregate_dict[key]

        for k in component.pnl.keys():
            if hasattr(df, "region1"):
                pnl[k] = (
                    component.pnl[k]
                    .groupby([df.region1, df.carrier, df.build_year], axis=1)
                    .agg(agg[k], **agg_group_kwargs)
                )

                pnl[k].columns = pd.Index([f"{region} {i}-{int(j)}" for region, i, j in pnl[k].columns])
            elif hasattr(df, "build_year"):
                pnl[k] = (
                    component.pnl[k]
                    .groupby([df.region, df.carrier, df.build_year], axis=1)
                    .agg(agg[k], **agg_group_kwargs)
                )
                pnl[k].columns = pd.Index([f"{region} {i}-{int(j)}" for region, i, j in pnl[k].columns])
            else:
                pnl[k] = (
                    component.pnl[k]
                    .groupby([df.region, df.carrier], axis=1)
                    .agg(agg[k], **agg_group_kwargs)
                )
                pnl[k].columns = pd.Index([f"{region} {i}" for region, i in pnl[k].columns])
            pnl[k].fillna(
                n.components[component.name]["attrs"].loc[k, "default"], inplace=True
            )

    # drop not needed components --------------------------------------------
    to_drop = ["H2 pipeline", "H2 pipeline retrofitted", "Gas pipeline", "DC"]
    to_drop = m.links[m.links.carrier.isin(to_drop)].index
    m.mremove("Link", to_drop)
    # TODO
    # dac_i = m.links[m.links.carrier == "DAC"].index
    # if snakemake.config["cluster_heat_nodes"]:
    #     m.links.loc[dac_i, "bus3"] = "urban decentral heat"
    # else:
    #     m.links.loc[dac_i, "bus3"] = "services urban decentral heat"
    # drop old global constraints
    m.global_constraints.drop(m.global_constraints.index, inplace=True)

    # add AC lines
    for component in n.iterate_components(["Line", "Link"]):
        df = component.df
        if component.name=="Link":
            df = df[df.carrier=="DC"]
        # drop grid in region
        df = df[df.region1!=df.region]
        # order to avoid multiple lines between regions
        positive_order = df.region < df.region1
        df_p = df[positive_order]
        swap_regions = {"region": "region1", "region1": "region",
                        "bus0": "bus1", "bus1": "bus0"}
        df_n = df[~positive_order].rename(columns=swap_regions)
        df = pd.concat([df_p, df_n])

        default = n.components[component.name]["attrs"]["default"]
        for col in df.columns.intersection(default.index):
            df[col].fillna(default.loc[col], inplace=True)

        keys = df.columns.intersection(aggregate_dict.keys())
        agg = dict(
            zip(
                df.columns.difference(keys),
                ["first"] * len(df.columns.difference(keys)),
            )
        )
        for key in keys:
            agg[key] = aggregate_dict[key]

        df = df.groupby(["region", "region1"]).agg(agg, **agg_group_kwargs)
        df.index = pd.Index([f"{region}-{i}" for region, i in df.index])
        # rename buses
        df.loc[:, df.columns.str.contains("bus")] = df.loc[
            :, df.columns.str.contains("bus")
        ].apply(lambda x: x.map(n.buses.region) + " " + x.map(n.buses.carrier))
        # drop the standard types to avoid them being read in twice
        if component.name in n.standard_type_components:
            df = component.df.drop(m.components[component.name]["standard_types"].index)

        import_components_from_dataframe(m, df, component.name)


    return m
    # ---------------------------------------------------------------------


def cluster_network(n, years):
    """Cluster network n to network m with one representative node."""
    logger.info("Cluster network spatially to one representative node.")
    assign_location(n)
    # clustered network m
    m = pypsa.Network(override_component_attrs=override_component_attrs)
    # set snapshots
    m.set_snapshots(n.snapshots)
    m.snapshot_weightings = n.snapshot_weightings.copy()
    m.investment_period_weightings = n.investment_period_weightings.copy()

    # catch all remaining attributes of network
    for attr in ["name", "srid"]:
        setattr(m, attr, getattr(n, attr))

    other_comps = sorted(n.all_components - {"Bus", "Carrier"} - {"Line"})
    # overwrite static component attributes
    for component in n.iterate_components(["Bus", "Carrier"] + other_comps):
        df = component.df
        default = n.components[component.name]["attrs"]["default"]
        for col in df.columns.intersection(default.index):
            df[col].fillna(default.loc[col], inplace=True)
        if hasattr(df, "carrier"):
            keys = df.columns.intersection(aggregate_dict.keys())
            agg = dict(
                zip(
                    df.columns.difference(keys),
                    ["first"] * len(df.columns.difference(keys)),
                )
            )
            for key in keys:
                agg[key] = aggregate_dict[key]
            if hasattr(df, "build_year"):
                df = df.groupby(["carrier", "build_year"]).agg(agg, **agg_group_kwargs)
                df.index = pd.Index([f"{i}-{int(j)}" for i, j in df.index])
            else:
                df = df.groupby("carrier").agg(agg, **agg_group_kwargs)
            # rename location
            df["country"] = "EU"
            df["location"] = "EU"
            # df["carrier"] = df.index
            # rename buses
            df.loc[:, df.columns.str.contains("bus")] = df.loc[
                :, df.columns.str.contains("bus")
            ].apply(lambda x: x.map(n.buses.carrier))
        # drop the standard types to avoid them being read in twice
        if component.name in n.standard_type_components:
            df = component.df.drop(m.components[component.name]["standard_types"].index)

        import_components_from_dataframe(m, df, component.name)

    # time varying data
    for component in n.iterate_components():
        pnl = getattr(m, component.list_name + "_t")
        df = component.df
        if not hasattr(df, "carrier"):
            continue
        keys = pd.Index(component.pnl.keys()).intersection(aggregate_dict.keys())
        agg = dict(
            zip(
                pd.Index(component.pnl.keys()).difference(aggregate_dict.keys()),
                ["first"]
                * len(pd.Index(component.pnl.keys()).difference(aggregate_dict.keys())),
            )
        )
        for key in keys:
            agg[key] = aggregate_dict[key]

        for k in component.pnl.keys():
            if hasattr(df, "build_year"):
                pnl[k] = (
                    component.pnl[k]
                    .groupby([df.carrier, df.build_year], axis=1)
                    .agg(agg[k], **agg_group_kwargs)
                )
                pnl[k].columns = pd.Index([f"{i}-{int(j)}" for i, j in pnl[k].columns])
            else:
                pnl[k] = (
                    component.pnl[k]
                    .groupby(df.carrier, axis=1)
                    .agg(agg[k], **agg_group_kwargs)
                )
            pnl[k].fillna(
                n.components[component.name]["attrs"].loc[k, "default"], inplace=True
            )

    # drop not needed components --------------------------------------------
    to_drop = ["H2 pipeline", "H2 pipeline retrofitted", "Gas pipeline", "DC"]
    to_drop = m.links[m.links.carrier.isin(to_drop)].index
    m.mremove("Link", to_drop)
    # TODO
    dac_i = m.links[m.links.carrier == "DAC"].index
    if snakemake.config["cluster_heat_nodes"]:
        m.links.loc[dac_i, "bus3"] = "urban decentral heat"
    else:
        m.links.loc[dac_i, "bus3"] = "services urban decentral heat"
    # drop old global constraints
    m.global_constraints.drop(m.global_constraints.index, inplace=True)

    # ---------------------------------------------------------------------
    # add representative generators for some countries and remove other res
    representative_ct = ["DE", "GR", "GB", "IT", "ES", "PT", "IE"]

    logger.info(
        "Take typical timeseries for renewable generators from the "
        "following representative countries {}".format(representative_ct)
    )
    split_carriers = [
        "offwind-ac",
        "onwind",
        "residential rural solar thermal",
        "residential urban decentral solar thermal",
        "services rural solar thermal",
        "services urban decentral solar thermal",
        "solar rooftop",
        "solar",
        "urban central solar thermal",
        "offwind",
        "offwind-dc",
    ]
    gens_i = n.generators[
        n.generators.carrier.isin(split_carriers)
        & (n.generators.country.isin(representative_ct))
    ].index
    split_df = n.generators.loc[gens_i]
    to_drop = m.generators[m.generators.carrier.isin(split_carriers)]
    # scale up p_nom and p_nom_max
    for attr in ["p_nom", "p_nom_max"]:
        attr_total = n.generators.groupby(["carrier", "build_year"]).sum(
            **agg_group_kwargs
        )[attr]
        attr_cts = split_df.groupby(["carrier", "build_year"]).sum(**agg_group_kwargs)[
            attr
        ]
        weight = attr_total.loc[attr_cts.index] / attr_cts
        default = n.components["Generator"]["attrs"]["default"].loc[attr]
        weight_series = (
            pd.Series(
                split_df.set_index(["carrier", "build_year"]).index.map(weight),
                index=split_df.index,
            )
            .loc[split_df.index]
            .fillna(default)
        )
        split_df[attr] = split_df[attr].mul(weight_series).fillna(default)

    split_df["bus"] = split_df.bus.map(n.buses.carrier)

    m.mremove("Generator", to_drop.index)
    import_components_from_dataframe(m, split_df, "Generator")

    m.generators_t.p_max_pu[split_df.index] = n.generators_t.p_max_pu[split_df.index]

    return m


def select_cts(n, years):
    """Cluster network n to network m with representative countries."""
    cluster_regions = {"GB": "UK"}
    countries = snakemake.config["select_cts"]
    logger.info("Consider only the following countries {}.".format(countries))
    assign_location(n)
    # clustered network m
    m = pypsa.Network(override_component_attrs=override_component_attrs)
    # set snapshots
    m.set_snapshots(n.snapshots)
    m.snapshot_weightings = n.snapshot_weightings.copy()
    m.investment_period_weightings = n.investment_period_weightings.copy()

    # catch all remaining attributes of network
    for attr in ["name", "srid"]:
        setattr(m, attr, getattr(n, attr))

    other_comps = sorted(n.all_components - {"Bus", "Carrier"})
    # overwrite static component attributes
    for component in n.iterate_components(["Bus", "Carrier"] + other_comps):
        df = component.df
        default = n.components[component.name]["attrs"]["default"]
        for col in df.columns.intersection(default.index):
            df[col].fillna(default.loc[col], inplace=True)
        if hasattr(df, "country"):
            keys = df.columns.intersection(aggregate_dict.keys())
            agg = dict(
                zip(
                    df.columns.difference(keys),
                    ["first"] * len(df.columns.difference(keys)),
                )
            )
            for key in keys:
                agg[key] = aggregate_dict[key]
            df = df[df.country.isin(countries + ["EU"])]
        if hasattr(df, "bus1"):
            df = df[df.bus1.isin(m.buses.index)]
        # drop the standard types to avoid them being read in twice
        if component.name in n.standard_type_components:
            df = component.df.drop(m.components[component.name]["standard_types"].index)

        import_components_from_dataframe(m, df, component.name)

    # time varying data
    for component in n.iterate_components():
        pnl = getattr(m, component.list_name + "_t")
        df = m.df(component.name)
        if not hasattr(df, "country"):
            continue

        for k in component.pnl.keys():
            pnl[k] = component.pnl[k].reindex(columns=df.index).dropna(axis=1)

    # drop not needed components --------------------------------------------
    to_drop = ["H2 pipeline retrofitted"]
    to_drop = m.links[m.links.carrier.isin(to_drop)].index
    m.mremove("Link", to_drop)
    # TODO DAC --------------------------------------------
    dac_i = m.links[m.links.carrier == "DAC"].index
    remove = m.links.loc[dac_i][~m.links.loc[dac_i, "bus3"].isin(m.buses.index)].index
    m.mremove("Link", remove)
    dac_i = m.links[m.links.carrier == "DAC"].index
    m.links.loc[dac_i, "bus3"] = m.links.loc[dac_i, "bus3"].str.replace(
        "urban central heat", "services urban decentral heat"
    )
    # drop old global constraints
    m.global_constraints.drop(m.global_constraints.index, inplace=True)

    # components which are previously for all countries ---------------------
    # TODO rename n -> m
    # land transport oil demand and emissions ###
    transport = pd.read_csv(snakemake.input.transport, index_col=0, parse_dates=True)
    buses = n.buses[n.buses.country.isin(countries) & (n.buses.carrier == "AC")].index
    share_of_total = transport[buses].sum(axis=1) / transport.sum(axis=1)
    land_transport_carriers = ["land transport oil", "land transport oil emissions"]
    land_transport_i = m.loads[m.loads.carrier.isin(land_transport_carriers)].index
    m.loads_t.p_set[land_transport_i] = n.loads_t.p_set[land_transport_i].mul(
        share_of_total.reindex(n.snapshots, level=1), axis=0
    )
    # biomass potentials ###
    biomass_potentials = pd.read_csv(snakemake.input.biomass_potentials, index_col=0)
    share_of_total = biomass_potentials.loc[countries].sum() / biomass_potentials.sum()

    biogas_i = n.stores[n.stores.bus == "EU biogas"].index
    m.stores.loc[biogas_i, ["e_nom", "e_initial"]] = n.stores.loc[
        biogas_i, ["e_nom", "e_initial"]
    ].mul(share_of_total["biogas"])

    biomass_i = n.stores[n.stores.bus == "EU solid biomass"].index
    m.stores.loc[biomass_i, ["e_nom", "e_initial"]] = n.stores.loc[
        biomass_i, ["e_nom", "e_initial"]
    ].mul(share_of_total["biogas"])

    # industrial demand ##
    industrial_demand = pd.read_csv(snakemake.input.industrial_demand, index_col=0)
    factor = (
        biomass_potentials.loc[countries, "solid biomass"].sum()
        / industrial_demand["solid biomass"].loc[buses].sum()
    )
    if factor > 1:
        logger.warning(
            "Solid biomass demand of industry of selected countries"
            " is larger than  solid biomass potential. Increasing "
            "biomass potentials to cover industry demand."
        )
        m.stores.loc[biomass_i, ["e_nom", "e_initial"]] *= factor
    # soild biomass for industry
    share_of_total = (
        industrial_demand["solid biomass"].loc[buses].sum()
        / industrial_demand["solid biomass"].sum()
    )
    m.loads_t.p_set["solid biomass for industry"] = (
        n.loads_t.p_set["solid biomass for industry"] * share_of_total
    )
    # gas for industry
    share_of_total = (
        industrial_demand["methane"].loc[buses].sum()
        / industrial_demand["methane"].sum()
    )
    m.loads_t.p_set["gas for industry"] = (
        n.loads_t.p_set["gas for industry"] * share_of_total
    )
    # naphta for industry
    share_of_total = (
        industrial_demand["naphtha"].loc[buses].sum()
        / industrial_demand["naphtha"].sum()
    )
    m.loads_t.p_set["naphtha for industry"] = (
        n.loads_t.p_set["naphtha for industry"] * share_of_total
    )

    # shipping ###
    all_navigation = ["total international navigation", "total domestic navigation"]
    nodal_energy_totals = pd.read_csv(snakemake.input.nodal_energy_totals, index_col=0)
    share_of_total = (
        nodal_energy_totals.loc[buses, all_navigation].sum().sum()
        / nodal_energy_totals[all_navigation].sum().sum()
    )
    m.loads_t.p_set["shipping oil emissions"] = (
        n.loads_t.p_set["shipping oil emissions"] * share_of_total
    )

    # aviation
    all_aviation = ["total international aviation", "total domestic aviation"]
    share_of_total = (
        nodal_energy_totals.loc[buses, all_aviation].sum().sum()
        / nodal_energy_totals[all_aviation].sum().sum()
    )
    m.loads_t.p_set["kerosene for aviation"] = (
        n.loads_t.p_set["kerosene for aviation"] * share_of_total
    )

    # oil emissions ##
    co2_release = ["naphtha for industry", "kerosene for aviation"]
    co2 = m.loads_t.p_set[co2_release].sum(axis=1) * 0.27 - (
        industrial_demand.loc[buses, "process emission from feedstock"].sum()
        / 1e6
        / 8760
    )
    m.loads_t.p_set["oil emissions"] = -co2

    # process emissions ##
    share_of_total = (
        industrial_demand.loc[
            buses, ["process emission", "process emission from feedstock"]
        ]
        .sum()
        .sum()
        / industrial_demand[["process emission", "process emission from feedstock"]]
        .sum()
        .sum()
    )
    m.loads_t.p_set["process emissions"] = (
        n.loads_t.p_set["process emissions"] * share_of_total
    )

    return m


def assign_location(n):
    """Assign locaion to buses, one port components."""
    n.buses["country"] = n.buses.rename(
        index=lambda x: x[:2] if (x[:2].isupper() and x not in ["AC", "H2"]) else "EU"
    ).index
    for c in n.one_port_components:
        n.df(c)["country"] = n.df(c).bus.map(n.buses.country)
    for c in n.branch_components:
        n.df(c)["country"] = n.df(c).bus0.map(n.buses.country)
        eu_i = n.df(c)["country"] == "EU"
        n.df(c).loc[eu_i, "country"] = n.df(c).loc[eu_i, "bus1"].map(n.buses.country)


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
    # assign countries to buses and one port components
    assign_location(n)

    # global installed capacities ------------------------------------
    global_capacity = pd.read_csv(snakemake.input.global_capacity, index_col=0).iloc[
        :, 0
    ]
    # rename inverter -> charger
    global_capacity.rename(index={"battery inverter": "battery charger"}, inplace=True)
    # rename pypsa-eur and sec different syntax
    global_capacity.loc["H2 Fuel Cell"] = global_capacity.loc["H2 fuel cell"]
    global_capacity.loc["H2 Electrolysis"] = global_capacity.loc["H2 electrolysis"]

    # local (per country) capacities ------------------------------------
    local_capacity = pd.read_csv(snakemake.input.local_capacity, index_col=0)
    if all(n.buses.country == "EU"):
        local_capacity["alpha_2"] = "EU"
    else:
        countries = n.generators.country.unique()
        local_capacity = local_capacity[local_capacity.alpha_2.isin(countries)]

    # for GlobalConstraint of the technical limit at each node, get the p_nom_max
    p_nom_max_limit = n.generators.p_nom_max.groupby(
        [n.generators.carrier, n.generators.bus, n.generators.build_year]
    ).sum(**agg_group_kwargs)
    p_nom_max_limit = p_nom_max_limit.xs(years[0], level=2)

    # global factor
    global_factor = (
        local_capacity.groupby(local_capacity.index)
        .sum()
        .div(global_capacity, axis=0)
        .fillna(gf_default)
        .iloc[:, 0]
    )

    return (global_capacity, p_nom_max_limit, global_factor)


def set_scenario_opts(n, opts):
    """Set scenario options."""
    for o in opts:
        # learning
        if "learn" in o:
            techs = list(snakemake.config["learning_rates"].keys())
            if any([tech in o.replace("x", " ") for tech in techs]):
                tech = max(
                    [tech for tech in techs if tech in o.replace("x", " ")], key=len
                )
                learn_diff = float(
                    o[len("learn" + tech) :].replace("p", ".").replace("m", "-")
                )
                learning_rate = snakemake.config["learning_rates"][tech] + learn_diff
                factor = global_factor.loc[tech]
                if "local" in opts:
                    factor = 1.0
                logger.info(
                    "technology learning for {} with learning rate {}%".format(
                        tech, learning_rate * 100
                    )
                )
                if tech not in n.carriers.index:
                    n.add("Carrier", name=tech)
                n.carriers.loc[tech, "learning_rate"] = learning_rate
                n.carriers.loc[tech, "global_capacity"] = global_capacity.loc[tech]
                n.carriers.loc[tech, "max_capacity"] = 30 * global_capacity.loc[tech]
                n.carriers.loc[tech, "global_factor"] = factor
                for c in nominal_attrs.keys():
                    ext_i = get_extendable_i(n, c)
                    if "carrier" not in n.df(c) or n.df(c).empty:
                        continue

                    if tech=="offwind":
                        rename_offwind = {
                            "offwind-dc": "offwind",
                            "offwind-ac": "offwind",
                        }
                        n.df(c).carrier.replace(rename_offwind, inplace=True)
                    learn_assets = n.df(c)[n.df(c)["carrier"] == tech].index
                    learn_assets = ext_i.intersection(
                        n.df(c)[n.df(c)["carrier"] == tech].index
                    )

                    if learn_assets.empty:
                        continue
                    index = (
                        n.df(c)
                        .loc[learn_assets][
                            n.df(c).loc[learn_assets, "build_year"] <= years[0]
                        ]
                        .index
                    )
                    # subtract the non-learning part of the offshore wind costs
                    if "nolearning_cost" in n.df(c).columns:
                        n.carriers.loc[tech, "initial_cost"] = (
                            n.df(c).loc[index, "capital_cost"]
                            - n.df(c).loc[index, "nolearning_cost"].fillna(0)
                        ).mean()
                    else:
                        n.carriers.loc[tech, "initial_cost"] = (
                            n.df(c).loc[index, "capital_cost"].mean()
                        )
                # TODO
                if tech == "H2 electrolysis":
                    n.carriers.loc[tech, "global_factor"]  = 0.4
                    # todo assume always local learning for H2 electrolysis
                    logger.info("assume local learning for H2 Electrolysis.")
                    n.carriers.loc[tech, "global_factor"]  = 1.
                    factor = 1.
                    # todo global capacity
                    n.carriers.loc[tech, "global_capacity"] = 1e3
                    n.carriers.loc["H2 electrolysis", "max_capacity"] = 5e6 / factor
                if tech == "H2 Electrolysis":
                    n.carriers.loc[tech, "global_factor"]  = 0.4
                    # todo assume always local learning for H2 electrolysis
                    logger.info("assume local learning for H2 Electrolysis.")
                    n.carriers.loc[tech, "global_factor"]  = 1.
                    factor = 1.
                    n.carriers.loc["H2 Electrolysis", "max_capacity"] = 6e6 / factor
                if tech == "H2 Fuel Cell":
                    n.carriers.loc["H2 Fuel Cell", "max_capacity"] = 2e4
                if tech == "DAC":
                    n.carriers.loc["DAC", "max_capacity"] = 200e3 / factor
                if tech == "solar":
                    n.carriers.loc["solar", "max_capacity"] = 3e6 / factor
                if tech == "onwind":
                    n.carriers.loc["onwind", "max_capacity"] = 3e6 / factor
                if tech == "offwind":
                    n.carriers.loc["offwind", "max_capacity"] = 3.5e6 / factor
                if tech == "battery":
                    bev_dsm_i = n.stores[n.stores.carrier == "battery storage"].index
                    n.stores.loc[bev_dsm_i, "e_nom_max"] = n.stores.loc[bev_dsm_i, "e_nom"]
                    n.stores.loc[bev_dsm_i, "e_nom_extendable"] = True
                    n.stores.loc[bev_dsm_i, "carrier"] = "battery"
        if "fcev" in o:
            fcev_fraction = float(o.replace("fcev", "")) / 100
            n = adjust_land_transport_share(n, fcev_fraction)

        if "co2seq" in o:
            factor = float(o.replace("co2seq", ""))
            n.stores.loc["co2 stored-{}".format(years[0]), "e_nom_max"] *= factor
            logger.info(
                "Total CO2 sequestration potential is set to {}".format(
                    n.stores.loc["co2 stored-{}".format(years[0]), "e_nom_max"]
                )
            )

        if "SMRc" in o:
            smr_i = n.links[n.links.carrier=="SMR CC"].index
            capital_cost0 = n.links.loc[smr_i, "capital_cost"].groupby(n.links.build_year).mean().loc[years[0]]
            factor = float(
                o[len("SMRc"):].replace("p", ".").replace("m", "-")
            )
            new_costs = pd.Series(np.interp(x=years, xp=[years[0],years[-1]],
                                            fp=[capital_cost0, factor * capital_cost0]),
                                  index=years)
            logger.info("updating SMR CC costs to \n "
                        "{}\n"
                        "------------------------------------------------\n".format(new_costs))
            n.links.loc[smr_i, "capital_cost"] = n.links.loc[smr_i, "build_year"].map(new_costs)

        if "H2Elc" in o:
            smr_i = n.links[n.links.carrier=="H2 Electrolysis"].index
            capital_cost0 = n.links.loc[smr_i, "capital_cost"].groupby(n.links.build_year).mean().loc[years[0]]
            factor = float(
                o[len("H2Elc"):].replace("p", ".").replace("m", "-")
            )
            new_costs = pd.Series(np.interp(x=years, xp=[years[0],years[-1]],
                                            fp=[capital_cost0, factor * capital_cost0]),
                                  index=years)
            logger.info("updating SH2 Electrolysis costs to \n "
                        "{}\n"
                        "------------------------------------------------\n".format(new_costs))
            n.links.loc[smr_i, "capital_cost"] = n.links.loc[smr_i, "build_year"].map(new_costs)


    if "local" in opts:
        learn_i = n.carriers[n.carriers.learning_rate != 0].index
        n.carriers.loc[learn_i, "global_factor"] = 1.0

    if "autonomy" in opts:
        logger.info("remove AC or DC connection between countries")
        remove = n.links[n.links.carrier == "DC"].index
        n.mremove("Link", remove)
        remove = n.lines.index
        n.mremove("Line", remove)

    return n


def set_temporal_aggregation(n, opts):
    """Aggregate network temporally."""
    for o in opts:
        # temporal clustering
        m = re.match(r"^\d+h$", o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m.group(0))
        # representive snapshots
        m = re.match(r"^\d+sn$", o, re.IGNORECASE)
        if m is not None:
            sn = int(m.group(0).split("sn")[0])
            n.set_snapshots(n.snapshots[::sn])
            n.snapshot_weightings *= sn
        # typical periods
        m = re.match(r"^\d+p\d\d+h", o, re.IGNORECASE)
        if m is not None:
            opts_t = snakemake.config["temporal_aggregation"]
            n_periods = int(o.split("p")[0])
            hours = int(o.split("p")[1].split("h")[0])
            clusterMethod = opts_t["clusterMethod"].replace("-", "_")
            extremePeriodMethod = opts_t["extremePeriodMethod"].replace("-", "_")
            if "Noneextreme" in opts:
                extremePeriodMethod = "None"
            if "kmedoids" in opts:
                clusterMethod = "k_medoids"
            kind = opts_t["kind"]

            logger.info(
                "\n------temporal clustering----------------------------\n"
                "aggregrate network to {} periods with length {} hours. \n"
                "Cluster method: {}\n"
                "extremePeriodMethod: {}\n"
                "optimisation kind: {}\n"
                "------------------------------------------------------\n".format(
                    n_periods, hours, clusterMethod, extremePeriodMethod, kind
                )
            )

            aggregate_snapshots(
                n,
                n_periods=n_periods,
                hours=hours,
                clusterMethod=clusterMethod,
                extremePeriodMethod=extremePeriodMethod,
                solver="gurobi_direct",
            )
    return n


def average_every_nhours(n, offset):
    """Temporal aggregate pypsa Network depending on offset."""
    logger.info("Resampling the network to {}".format(offset))
    m = n.copy(with_time=False)

    # fix copying of network attributes
    # copied from pypsa/io.py, should be in pypsa/components.py#Network.copy()
    allowed_types = (float, int, bool) + string_types + tuple(np.typeDict.values())
    attrs = dict(
        (attr, getattr(n, attr))
        for attr in dir(n)
        if (not attr.startswith("__") and isinstance(getattr(n, attr), allowed_types))
    )
    for k, v in iteritems(attrs):
        setattr(m, k, v)

    def resample_multi(df, offset, arg="sum"):
        resampled_df = pd.DataFrame()
        for year in df.index.levels[0]:
            year_df = df.loc[year].resample(offset).agg(arg)
            year_df.index = pd.MultiIndex.from_product(
                [[year], year_df.index], names=df.index.names
            )
            resampled_df = pd.concat([resampled_df, year_df])
        return resampled_df

    snapshot_weightings = resample_multi(n.snapshot_weightings, offset, "sum")
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in iteritems(c.pnl):
            if not df.empty:
                if c.list_name == "stores" and k == "e_max_pu":
                    pnl[k] = resample_multi(df, offset, "min")
                elif c.list_name == "stores" and k == "e_min_pu":
                    pnl[k] = resample_multi(df, offset, "max")
                else:
                    pnl[k] = resample_multi(df, offset, "mean")

    return m


def set_carbon_constraints(n):
    """Add global constraints for carbon emissions."""
    budget = (
        snakemake.config["co2_budget"]["1p7"] * 1e9
    )  # budget for + 1.5 Celsius for Europe
    for o in opts:
        # temporal clustering
        m = re.match(r"^\d+p\d$", o, re.IGNORECASE)
        if m is not None:
            budget = snakemake.config["co2_budget"][m.group(0)] * 1e9
    logger.info("add carbon budget of {}".format(budget))
    n.add(
        "GlobalConstraint",
        "Budget",
        type="Co2Budget",
        carrier_attribute="co2_emissions",
        sense="<=",
        investment_period=n.snapshots.levels[0][-1],
        constant=budget,
    )

    if not "noco2neutral" in opts:
        logger.info("Add carbon neutrality constraint.")
        n.add(
            "GlobalConstraint",
            "Co2neutral",
            type="Co2Neutral",
            carrier_attribute="co2_emissions",
            investment_period=n.snapshots.levels[0][-1],
            sense="<=",
            constant=0,
        )

    logger.info("add minimum emissions for {} ".format(n.snapshots.levels[0][0]))
    n.add(
        "GlobalConstraint",
        "Co2min",
        type="Co2min",
        carrier_attribute="co2_emissions",
        sense=">=",
        investment_period=n.snapshots.levels[0][0],
        constant=3.2e9,
    )


    # logger.info("add carbon budget of {} for 2030".format(0.12*budget))
    # n.add(
    #     "GlobalConstraint",
    #     "Co2target2030",
    #     type="Co2Target2",
    #     carrier_attribute="co2_emissions",
    #     sense="<=",
    #     investment_period=2030,
    #     constant=0.12*budget,
    # )

    # logger.info("add carbon budget of {} for 2040".format(0.15*budget))
    # n.add(
    #     "GlobalConstraint",
    #     "Co2target2040",
    #     type="Co2Target2",
    #     carrier_attribute="co2_emissions",
    #     sense="<=",
    #     investment_period=2040,
    #     constant=0.8*budget,
    # )
    # -------------------------------------------------------------------------

    if not "notarget" in opts:
        logger.info("add CO2 targets.")
        emissions_1990 = 4.53693
        emissions_2019 = 3.344096

        logger.info("add carbon target of {} Gt for 2020 to stay below 2019 emissions".format(emissions_2019))
        n.add(
            "GlobalConstraint",
            "CarbonTarget2020",
            type="Co2Target",
            carrier_attribute="co2_emissions",
            sense="<=",
            investment_period=2020,
            constant=emissions_2019*1e9,
        )

        logger.info("add carbon target of {} Gt for 2030".format(0.45*emissions_1990))
        n.add(
            "GlobalConstraint",
            "CarbonTarget2030",
            type="Co2Target",
            carrier_attribute="co2_emissions",
            sense="<=",
            investment_period=2030,
            constant=0.45*emissions_1990*1e9,
        )

        if 2035 in n.investment_period_weightings.index:
            logger.info("add carbon target of {} Gt for 2035".format(0.3*emissions_1990))
            n.add(
                "GlobalConstraint",
                "CarbonTarget2035",
                type="Co2Target",
                carrier_attribute="co2_emissions",
                sense="<=",
                investment_period=2035,
                constant=0.3*emissions_1990*1e9,
            )


        logger.info("add carbon target of {} Gt for 2040".format(0.1*emissions_1990))
        n.add(
            "GlobalConstraint",
            "CarbonTarget2040",
            type="Co2Target",
            carrier_attribute="co2_emissions",
            sense="<=",
            investment_period=2040,
            constant=0.1*emissions_1990*1e9,
        )

        if 2045 in n.investment_period_weightings.index:
            logger.info("add carbon target of {} Gt for 2045".format(0.05*emissions_1990))
            n.add(
                "GlobalConstraint",
                "CarbonTarget2045",
                type="Co2Target",
                carrier_attribute="co2_emissions",
                sense="<=",
                investment_period=2045,
                constant=0.05*emissions_1990*1e9,
            )
    return n


def set_max_growth(n):
    """Limit build rate of renewables."""
    logger.info("set maximum growth rate of renewables.")
    # solar max grow so far 28 GW in Europe https://www.iea.org/reports/renewables-2020/solar-pv
    n.carriers.loc["solar", "max_growth"] = 90 * 1e3  # 70 * 1e3
    # onshore max grow so far 16 GW in Europe https://www.iea.org/reports/renewables-2020/wind
    n.carriers.loc["onwind", "max_growth"] = 60 * 1e3  # 40 * 1e3
    # offshore max grow so far 3.5 GW in Europe https://windeurope.org/about-wind/statistics/offshore/european-offshore-wind-industry-key-trends-statistics-2019/
    n.carriers.loc[["offwind-ac", "offwind-dc"], "max_growth"] = 15 * 1e3  # 8.75 * 1e3

    return n


def set_min_growth(n):
    """Limit build rate of renewables."""
    logger.info("set minimum growth rate of renewables.")
    # solar max grow so far 28 GW in Europe https://www.iea.org/reports/renewables-2020/solar-pv
    n.carriers.loc["solar", "min_growth"] = 5 * 1e3  # 70 * 1e3
    # onshore max grow so far 16 GW in Europe https://www.iea.org/reports/renewables-2020/wind
    n.carriers.loc["onwind", "min_growth"] = 5 * 1e3  # 40 * 1e3
    # EU commissio target for 2030 40 GW of H2 Electrolysis
    n.carriers.loc["H2 Electrolysis", "min_growth"] = 4 * 1e3
    # offshore max grow so far 3.5 GW in Europe https://windeurope.org/about-wind/statistics/offshore/european-offshore-wind-industry-key-trends-statistics-2019/
    n.carriers.loc[["offwind-ac", "offwind-dc"], "min_growth"] = 0.8 * 1e3  # 8.75 * 1e3

    return n


def adjust_land_transport_share(n, fcev_fraction=0.4):
    """Set shares of FCEV and EV.

    Redefines share of land transport for fuel cell cares (FCEV) and electric
    vehices (EV). This function is only for testing and will be removed later.
    """
    logger.info("Change fuel cell share in land transport to {}".format(fcev_fraction))

    # default land transport assumptions
    default_share = pd.DataFrame(
        np.array([[0, 0.05, 0.1, 0.15], [0, 0.25, 0.6, 0.85]]).T,
        index=[2020, 2030, 2040, 2050],
        columns=["land_transport_fuel_cell_share", "land_transport_electric_share"],
    )

    new_share = pd.concat(
        [
            default_share.sum(axis=1) * fcev_fraction,
            default_share.sum(axis=1) * (1 - fcev_fraction),
        ],
        axis=1,
    )
    new_share.columns = default_share.columns

    factor = new_share.div(default_share).fillna(0.0)

    # adjust to new shares
    to_change = [
        ("Load", "p_set", "land transport EV", "land_transport_electric_share"),
        ("Link", "p_nom", "BEV charger", "land_transport_electric_share"),
        ("Link", "p_nom", "V2G", "land_transport_electric_share"),
        ("Store", "e_nom", "battery storage", "land_transport_electric_share"),
        ("Load", "p_set", "land transport fuel cell", "land_transport_fuel_cell_share"),
    ]
    for (c, attr, carrier, share_type) in to_change:
        change_i = n.df(c)[n.df(c).carrier == carrier].index
        if c != "Load":
            n.df(c).loc[change_i, attr] *= (
                n.df(c).loc[change_i, "build_year"].map(factor[share_type])
            )
        else:
            load_w = factor[share_type].reindex(n.snapshots, level=0)
            n.pnl(c)["p_set"][change_i] = n.pnl(c)["p_set"][change_i].mul(
                load_w, axis=0
            )

    return n


# constraints ---------------------------------------------------------------
def add_battery_constraints(n):
    chargers = n.links.index[
        n.links.carrier.str.contains("battery charger") & n.links.p_nom_extendable
    ]
    dischargers = chargers.str.replace("charger", "discharger")
    if chargers.empty:
        return
    link_p_nom = get_var(n, "Link", "p_nom")

    lhs = linexpr(
        (1, link_p_nom[chargers]),
        (
            -n.links.loc[dischargers, "efficiency"].values,
            link_p_nom[dischargers].values,
        ),
    )

    define_constraints(n, lhs, "=", 0, "Link", "charger_ratio")


def add_carbon_neutral_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "Co2Neutral"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        rhs = glc.constant
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        time_valid = int(glc.loc["investment_period"])
        if not stores.empty:
            final_e = get_var(n, "Store", "e").groupby(level=0).last()[stores.index]
            lhs = linexpr(
                (-1, final_e.shift().loc[time_valid]), (1, final_e.loc[time_valid])
            )
            # define_constraints(n, lhs, "==", rhs, "GlobalConstraint", "Co2Neutral",
            #                    axes=pd.Index([name]))
            con = write_constraint(n, lhs, "<=", rhs, axes=pd.Index([name]))
            set_conref(n, con, "GlobalConstraint", "mu", name)


def add_carbon_budget_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "Co2Budget"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        rhs = glc.constant
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]
        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            time_valid = int(glc.loc["investment_period"])
            final_e = get_var(n, "Store", "e").groupby(level=0).last()[stores.index]

            #todo
            time_weightings = n.investment_period_weightings.time_weightings.mean()
            lhs = linexpr((time_weightings, final_e.loc[time_valid]))
            con = write_constraint(n, lhs, "<=", rhs, axes=pd.Index([name]))
            set_conref(n, con, "GlobalConstraint", "mu", name)



def add_carbon_minimum_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "Co2min"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        rhs = glc.constant
        sense = glc.sense
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]
        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            time_valid = int(glc.loc["investment_period"])
            final_e = get_var(n, "Store", "e").groupby(level=0).last()[stores.index]

            #todo

            lhs = linexpr((1, final_e.loc[time_valid]))
            con = write_constraint(n, lhs, sense, rhs, axes=pd.Index([name]))
            set_conref(n, con, "GlobalConstraint", "mu", name)


def add_carbon_target(n, snapshots):
    glcs = n.global_constraints.query('type == "Co2Target"')

    for name, glc in glcs.iterrows():
        rhs = glc.constant
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]
        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            time_valid = int(glc.loc["investment_period"])
            final_e = get_var(n, "Store", "e").groupby(level=0).last()[stores.index]
            first_e = get_var(n, "Store", "e").groupby(level=0).first()[stores.index]
            lhs = linexpr((1, final_e.loc[time_valid]),
                          (-1, first_e.loc[time_valid]))
            con = write_constraint(n, lhs, "<=", rhs, axes=pd.Index([name]))
            set_conref(n, con, "GlobalConstraint", "mu", name)

    glcs = n.global_constraints.query('type == "Co2Target2"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        rhs = glc.constant
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]
        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            time_valid = int(glc.loc["investment_period"])
            first_e = get_var(n, "Store", "e").groupby(level=0).first()[stores.index]
            lhs = linexpr((1, first_e.loc[time_valid]))
            con = write_constraint(n, lhs, "<=", rhs, axes=pd.Index([name]))
            set_conref(n, con, "GlobalConstraint", "mu", name)


def add_local_res_constraint(n, snapshots):
    c, attr = "Generator", "p_nom"
    res = ["offwind-ac", "offwind-dc", "onwind", "solar", "solar rooftop"]
    n.df(c)["country"] = (
        n.df(c).rename(lambda x: x[:2] if x[:2].isupper() else "EU").index
    )
    ext_i = n.df(c)[
        (n.df(c)["carrier"].isin(res))
        & (n.df(c)["country"] != "EU")
        & (n.df(c)["p_nom_extendable"])
    ].index
    time_valid = snapshots.levels[0]

    active_i = pd.concat(
        [
            get_active_assets(n, c, inv_p, snapshots).rename(inv_p)
            for inv_p in time_valid
        ],
        axis=1,
    ).astype(int)

    ext_and_active = active_i.T[active_i.index.intersection(ext_i)]

    if ext_and_active.empty:
        return

    cap_vars = get_var(n, c, attr)[ext_and_active.columns]

    lhs = (
        linexpr((ext_and_active, cap_vars))
        .T.groupby([n.df(c).carrier, n.df(c).country])
        .sum(**agg_group_kwargs)
        .T
    )

    p_nom_max_w = n.df(c).p_nom_max.loc[ext_and_active.columns]
    p_nom_max_t = expand_series(p_nom_max_w, time_valid).T

    rhs = (
        p_nom_max_t.mul(ext_and_active)
        .groupby([n.df(c).carrier, n.df(c).country], axis=1)
        .max(**agg_group_kwargs)
    ).reindex(columns=lhs.columns)

    define_constraints(n, lhs, "<=", rhs, "GlobalConstraint", "res_limit")


def add_capacity_constraint(n, snapshots):
    """Additional constraint for overall installed capacity to speed up optimisation."""
    logger.info("add constraint for minimum installed capacity to speed up optimisation")

    # c, attr = "Generator", "p_nom"
    # res = ["offwind", "onwind", "solar", "solar rooftop"]
    # ext_i = n.df(c)[
    #     (n.df(c)["carrier"].isin(res)) & (n.df(c)["p_nom_extendable"])
    # ].index

    # cap_vars = get_var(n, c, attr)[ext_i]

    # lhs = (
    #     linexpr((1, cap_vars))
    #     .groupby(n.df(c).carrier)
    #     .sum(**agg_group_kwargs)
    #     .reindex(index=res)
    # )

    # rhs = pd.Series([1800e3, 100e3, 1300e3, 100e3], index=res)

    # define_constraints(n, lhs, ">=", rhs, "GlobalConstraint", "min_cap")

    c, attr = "Link", "p_nom"
    res = ["H2 Electrolysis"]
    ext_i = n.df(c)[
        (n.df(c)["carrier"].isin(res)) & (n.df(c)["p_nom_extendable"])
    ].index

    cap_vars = get_var(n, c, attr)[ext_i]

    lhs = (
        linexpr((1, cap_vars))
        .groupby(n.df(c).carrier)
        .sum(**agg_group_kwargs)
        .reindex(index=res)
    )

    rhs = pd.Series([1e6], index=res)

    define_constraints(n, lhs, ">=", rhs, "Carrier", "min_cap")


def add_retrofit_gas_boilers_constraint(n, snapshots):
    """Allow retrofitting of existing gas boilers to H2 boilers"""
    logger.info("Add constraint for retrofitting gas boilers to H2 boilers.")
    c, attr = "Link", "p"

    h2_i = n.df(c)[n.df(c).carrier.str.contains("H2 boiler")].index
    gas_boiler_i = n.df(c)[
        n.df(c).carrier.str.contains("gas boiler") & n.df(c).p_nom_extendable
    ].index

    # TODO
    n.df(c).loc[h2_i, "capital_cost"] = 100.0

    h2_p = get_var(n, c, attr)[h2_i]
    gas_boiler_p = get_var(n, c, attr)[gas_boiler_i]

    gas_boiler_cap = get_var(n, c, "p_nom").loc[gas_boiler_i]
    gas_boiler_cap_t = expand_series(gas_boiler_cap, snapshots).T

    if snakemake.config["heat_must_run"]:
        n.links_t.p_max_pu.drop(h2_i.union(gas_boiler_i), axis=1, inplace=True,
                                errors="ignore")
        n.links_t.p_min_pu.drop(h2_i.union(gas_boiler_i), axis=1, inplace=True,
                                errors="ignore")
        # heat profile
        cols = n.loads_t.p_set.columns[n.loads_t.p_set.columns.str.contains("heat")
                                       & ~n.loads_t.p_set.columns.str.contains("industry")]
        profile = n.loads_t.p_set[cols] / n.loads_t.p_set[cols].groupby(level=0).max()
        profile.rename(columns=n.loads.bus.to_dict(), inplace=True)

        profile1 = profile.reindex(columns=n.links.bus1)
        profile1.columns = n.links.index
        profile1 = profile1[gas_boiler_i]

        lhs = linexpr((-1, h2_p),
                      (profile1.values, gas_boiler_cap_t.values),
                      (-1, gas_boiler_p.values))


    else:
        lhs = linexpr((-1, h2_p), (1, gas_boiler_cap_t.values), (-1, gas_boiler_p.values))
    rhs = 0.0
    define_constraints(n, lhs, ">=", rhs, "Link", "retro_gasboiler")
    if snakemake.config["heat_must_run"]:
        define_constraints(n, lhs, "<=", rhs, "Link", "retro_gasboiler_lb")

def add_retrofit_OCGT_constraint(n, snapshots):
    """Allow retrofitting of existing gas boilers to H2 boilers"""
    logger.info("Add constraint for retrofitting existing OCGT to run with H2.")
    c, attr = "Link", "p"

    h2_i = n.df(c)[n.df(c).carrier=="OCGT H2"].index
    gas_boiler_i = n.df(c)[(n.df(c).carrier=="OCGT") & (n.df(c).p_nom_extendable)].index

    # TODO
    n.df(c).loc[h2_i, "capital_cost"] = 100.0

    h2_p = get_var(n, c, attr)[h2_i]
    gas_boiler_p = get_var(n, c, attr)[gas_boiler_i]

    gas_boiler_cap = get_var(n, c, "p_nom").loc[gas_boiler_i]
    gas_boiler_cap_t = expand_series(gas_boiler_cap, snapshots).T

    lhs = linexpr((-1, h2_p), (1, gas_boiler_cap_t.values), (-1, gas_boiler_p.values))
    rhs = 0.0
    define_constraints(n, lhs, ">=", rhs, "Link", "retrofit_OCGT")


# --------------------------------------------------------------------
def prepare_network(n, solve_opts=None):
    """Add solving options to the network before calling the lopf.

    Solving options (solve_opts (type dict)), keys which effect the network in
    this function:

         'load_shedding'
         'noisy_costs'
         'clip_p_max_pu'
         'n_hours'
    """

    if any(n.carriers.learning_rate != 0) and not "seqcost" in opts:

        def extra_functionality(n, snapshots):
            add_battery_constraints(n)
            add_learning(
                n,
                snapshots,
                segments=snakemake.config["segments"],
                time_delay=time_delay, #snakemake.config["time_delay"],
            )
            add_carbon_neutral_constraint(n, snapshots)
            add_carbon_budget_constraint(n, snapshots)
            add_carbon_minimum_constraint(n, snapshots)
            add_carbon_target(n, snapshots)
            add_local_res_constraint(n, snapshots)
            if snakemake.config["h2boiler_retrofit"]:
                add_retrofit_gas_boilers_constraint(n, snapshots)
            if snakemake.config["OCGT_retrofit"]:
                add_retrofit_OCGT_constraint(n, snapshots)
            if snakemake.config["capacity_constraint"]:
                add_capacity_constraint(n, snapshots)

        skip_objective = True
        keep_shadowprices = False
    else:

        def extra_functionality(n, snapshots):
            add_battery_constraints(n)
            add_carbon_neutral_constraint(n, snapshots)
            add_carbon_minimum_constraint(n, snapshots)
            add_carbon_target(n, snapshots)
            add_carbon_budget_constraint(n, snapshots)
            add_local_res_constraint(n, snapshots)
            if snakemake.config["h2boiler_retrofit"]:
                add_retrofit_gas_boilers_constraint(n, snapshots)
            if snakemake.config["OCGT_retrofit"]:
                add_retrofit_OCGT_constraint(n, snapshots)
            if snakemake.config["capacity_constraint"]:
                add_capacity_constraint(n, snapshots)

        skip_objective = False
        keep_shadowprices = True

    # check for typcial periods
    if hasattr(n, "cluster"):
        typical_period = True
        # TODO
        typical_period = False
    else:
        typical_period = False

    config = snakemake.config["solving"]
    solve_opts = config["options"]
    solver_options = config["solver"].copy()
    solver_options["threads"] = snakemake.threads

    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if solve_opts.get("load_shedding"):
        n.add("Carrier", "Load")
        n.madd(
            "Generator",
            n.buses.index,
            " load",
            bus=n.buses.index,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=1e2,  # Eur/kWh
            # intersect between macroeconomic and surveybased
            # willingness to pay
            # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                np.random.seed(174)
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )
                if "carrier" in t.df and not t.df.empty and t.name!="Load":
                    learn_i = n.carriers[n.carriers.learning_rate != 0].index
                    ext_i = get_extendable_i(n, t.name)
                    learn_assets = t.df[t.df["carrier"].isin(learn_i)].index
                    learn_assets = ext_i.intersection(
                        t.df[t.df["carrier"].isin(learn_i)].index
                    )
                    if learn_assets.empty:
                        continue
                    nolearn_noise = 1 + 2 * (
                        np.random.random(len(t.df.loc[learn_assets])) - 0.5
                    )
                    if "nolearning_cost" not in t.df.columns:
                        t.df["nolearning_cost"] = np.NaN
                    t.df.loc[learn_assets, "nolearning_cost"] = t.df.loc[learn_assets, "nolearning_cost"].fillna(0) + nolearn_noise


        for t in n.iterate_components(["Line", "Link"]):
            np.random.seed(123)
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if "MIP1" in opts:
        solver_options["MIPFocus"] = 1
    if "MIP2" in opts:
        solver_options["MIPFocus"] = 2
    if "MIP3" in opts:
        solver_options["MIPFocus"] = 3

    return (
        extra_functionality,
        skip_objective,
        typical_period,
        solver_options,
        keep_shadowprices,
        n,
    )


# -----------------------------------------------------------------------
def seqlopf(
    n,
    min_iterations=4,
    max_iterations=6,
    track_iterations=False,
    msq_threshold=0.05,
    extra_functionality=None,
    time_delay=True,
):
    """
    Iterative linear optimization updating the capital costs according to learning
    curves. After each sucessful solving, investment costs are recalculated
    based on the optimization result. If warmstart is possible, it uses the
    result from the previous iteration to fasten the optimization.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    msq_threshold: float, default 0.05
        Maximal mean square difference between optimized investment costs of
        the current and the previous iteration. As soon as this threshold is
        undercut, and the number of iterations is bigger than 'min_iterations'
        the iterative optimization stops
    min_iterations : integer, default 4
        Minimal number of iteration to run regardless whether the msq_threshold
        is already undercut
    max_iterations : integer, default 6
        Maximal number of iterations to run regardless whether msq_threshold
        is already undercut
    track_iterations: bool, default False
        If True, the intermediate capital costs and values of the
        objective function are recorded for each iteration. The values of
        iteration 0 represent the initial state.
    **kwargs
        Keyword arguments of the lopf function which runs at each iteration
    """

    # -- helpers ----------------------------------
    def update_costs_of_learning_carriers(n):
        """Update the capital costs of all learning technologies.

        According to build year and carrier, taking the DEA cost assumptions
        of the respective year.
        """
        logger.info(
            "Update capital costs for learning technologies according to DEA assumptions at build year."
        )
        periods = n.investment_period_weightings.index
        cost_folder = snakemake.input.costs
        discount_rate = snakemake.config["costs"]["discountrate"]
        lifetime = snakemake.config["costs"]["lifetime"]
        costs = prepare_costs_all_years(
            periods, True, cost_folder, discount_rate, lifetime
        )
        learn_map = get_learn_assets_map(n)

        # overwrite cost assumptions
        for c in learn_map.keys():
            assets = learn_map[c].index
            # rename carriers according to names in tech data base
            learn_map[c].carrier.replace(pypsa_to_techbase, inplace=True)
            capital_cost_i = learn_map[c].set_index(["build_year", "carrier"]).index
            capital_costs_DEA = (
                pd.concat(costs).reindex(capital_cost_i).set_index(assets)["fixed"]
            )
            n.df(c).loc[assets, "capital_cost"] = capital_costs_DEA

        return n

    def get_new_cost(n, c, time_delay=True):
        """Calculate new cost depending on installed capacities."""
        # PARAMETERS #############################################
        assets = n.df(c).loc[prev[c].index]
        learn_carriers = assets.carrier.unique()
        # learning rate
        learning_rate = n.carriers.loc[learn_carriers, "learning_rate"]
        # initial installed capacity
        initial_cum_cap = n.carriers.loc[learn_carriers, "global_capacity"]
        # initial cost
        c0 = n.carriers.loc[learn_carriers, "initial_cost"]

        # CAPACITY ##########################################################
        # installed capacity per investment period
        attr = nominal_attrs[c]
        p_nom = assets.groupby([assets.carrier, assets.build_year]).sum()[f"{attr}_opt"]
        # cumulative installed global capacity
        cum_p_nom = (
            p_nom.unstack()
            .cumsum(axis=1)
            .div(n.carriers.loc[learn_carriers, "global_factor"], axis=0)
            .add(initial_cum_cap, axis=0)
        )
        cum_p_nom[0] = initial_cum_cap
        cum_p_nom.sort_index(axis=1, inplace=True)
        cumulative_cost = pd.DataFrame().reindex_like(cum_p_nom)

        # COST ############################################################################################################
        # cumulative cost according exactly to learning curve
        for carrier in learn_carriers:
            cumulative_cost.loc[carrier] = cum_p_nom.loc[carrier].apply(lambda x:
                                              cumulative_cost_curve(x,
                                                                    learning_rate.loc[carrier],
                                                                    c0.loc[carrier],
                                                                    initial_cum_cap.loc[carrier]))
        # (a) average cost paid per MW -> exaclty on learning curve (ATTENTION: this is NOT the same as investment costs (c)!)
        diff = cum_p_nom.diff(axis=1)
        diff.where(diff>5, other=0., inplace=True)
        cost_av = round(cumulative_cost.diff(axis=1)).div(diff.replace(0, np.nan))
        if (cost_av>expand_series(c0, cost_av.columns)).any(axis=None):
            breakpoint()
            tech_wrong = cost_av.index[(cost_av>expand_series(c0, cost_av.columns)).any(axis=1)]
            year_wrong = cost_av.columns[(cost_av>expand_series(c0, cost_av.columns)).any(axis=0)]
            logger.warning("average costs larger than c0 for {} \n  cum_p_nom {} \n cost av {} \n c0 {}".format(c, cum_p_nom.loc[tech_wrong, year_wrong], cost_av.loc[tech_wrong, year_wrong], c0.loc[tech_wrong]))
            cost_av.loc[tech_wrong, year_wrong] = c0.loc[tech_wrong]

        cost_av[0] = c0
        cost_av = cost_av.fillna(method="ffill", axis=1)
        if (cost_av.diff(axis=1)>0).any(axis=None):
            breakpoint()

        # (b) investment costs by piecewise linearisation
        x_high = n.carriers.loc[learn_carriers, "max_capacity"]
        segments = snakemake.config["segments"]
        points = get_linear_interpolation_points(n, initial_cum_cap, x_high, segments)
        slope = get_slope(points)
        # costs for new build capacity in investment period
        cost_plinear = pd.DataFrame(columns=learn_carriers)
        for carrier in learn_carriers:
            x_fit = points.xs("x_fit", level=1, axis=1)[carrier]
            cost_plinear[carrier] = (
                cum_p_nom.loc[carrier]
                .apply(lambda x: np.searchsorted(x_fit, x, side="right") - 1)
                .map(slope[carrier])
                .fillna(slope[carrier].iloc[-1])
            )
        if time_delay:
            cost_av = cost_av.shift(axis=1)
            cost_av.iloc[:,0] = c0

            cost_plinear = cost_plinear.shift()
            cost_plinear.iloc[0,:] = c0

        # (c) investment costs exactly form learning curve
        capital_cost = cum_p_nom.apply(
            lambda x: experience_curve(
                x,
                learning_rate.loc[x.name],
                c0.loc[x.name],
                initial_cum_cap.loc[x.name],
            ),
            axis=1,
        )


        # averaged paid costs
        cost_av = pd.Series(
            n.df(c).set_index(["carrier", "build_year"]).index.map(cost_av.stack()),
            index=n.df(c).index,
        ).fillna(n.df(c).capital_cost)
        # costs for newly build asset in investment period
        cost_plinear = pd.Series(
            n.df(c).set_index(["carrier", "build_year"]).index.map(cost_plinear.T.stack()),
            index=n.df(c).index,
        ).fillna(n.df(c).capital_cost)
        # add costs without learning
        if "nolearning_cost" in n.df(c).columns:
            cost_plinear.loc[assets.index] += n.df(c).loc[assets.index, "nolearning_cost"].fillna(0)
            cost_av.loc[assets.index] += n.df(c).loc[assets.index, "nolearning_cost"].fillna(0)
        return cost_av

    def update_cost_params(n, prev, time_delay):
        """Update cost parameters according to installed capacities."""
        for c in prev.keys():

            # overwrite cost assumptions
            n.df(c)["capital_cost"] = get_new_cost(n, c, time_delay)

    def msq_diff(n, prev):
        """Compare previous investment costs with current."""
        mean_square = 0
        for c in prev.keys():
            assets = n.df(c).loc[prev[c].index]
            err = (
                np.sqrt((prev[c].capital_cost - assets.capital_cost).pow(2).mean())
                / assets.capital_cost.mean()
            )
            mean_square = max(mean_square, err)
        logger.info(
            f"Maximum mean square difference after iteration {iteration} is "
            f"{mean_square}"
        )
        return mean_square

    def save_capital_cost(n, prev, iteration, status):
        """Save optmised capacities of each iteration step."""
        for c in prev.keys():
            n.df(c)["capital_cost_{}".format(iteration)] = n.df(c)["capital_cost"]
        setattr(n, f"status_{iteration}", status)
        setattr(n, f"objective_{iteration}", n.objective)
        n.iteration = iteration
        n.global_constraints["mu_{}".format(iteration)] = n.global_constraints["mu"]

    def get_learn_assets_map(n):
        """Return dictionary mapping component name -> learn assets."""
        learn_i = n.carriers[n.carriers.learning_rate != 0].index
        map_learn_assets = {}
        for c in n.branch_components | n.one_port_components:
            if "carrier" not in n.df(c) or n.df(c).empty or c == "Load":
                continue
            ext_i = get_extendable_i(n, c)
            learn_assets = n.df(c)[n.df(c)["carrier"].isin(learn_i)].index
            learn_assets = ext_i.intersection(
                n.df(c)[n.df(c)["carrier"].isin(learn_i)].index
            )
            if learn_assets.empty:
                continue
            map_learn_assets[c] = n.df(c).loc[learn_assets]
        return map_learn_assets

    # ----------------------------
    iteration = 1
    store_basis = True
    diff = msq_threshold

    # update capital costs of learning technologies according to build year
    if snakemake.config["seqcost_update"]:
        n = update_costs_of_learning_carriers(n)

    while diff >= msq_threshold or iteration < min_iterations:
        if iteration > max_iterations:
            logger.info(
                f"Iteration {iteration} beyond max_iterations "
                f"{max_iterations}. Stopping ..."
            )
            break
        # save previous cost and capacities
        prev = get_learn_assets_map(n)
        # run lopf
        warmstart = bool(iteration and ("basis_fn" in n.__dir__()))
        status, termination_condition = n.lopf(
            pyomo=False,
            warmstart=warmstart,
            store_basis=store_basis,
            solver_name="gurobi",
            skip_objective=skip_objective,
            multi_investment_periods=True,
            solver_options=solver_options,
            solver_logfile=snakemake.log.solver,
            extra_functionality=extra_functionality,
            typical_period=typical_period,
        )

        assert status == "ok", (
            f"Optimization failed with status {status}"
            f"and termination {termination_condition}"
        )
        # track capital costs of each iteration
        if track_iterations:
            save_capital_cost(n, prev, iteration, status)
        # update capital costs depending on installed capacities
        update_cost_params(n, prev, time_delay)
        # calculate difference between previous and current result
        diff = msq_diff(n, prev)
        iteration += 1


def remove_techs_for_speed(n):
    """remove certain technologies to speed up optimisation."""
    # remove completly
    remove = ["home battery", "home battery charger", "home battery discharger",
              "helmeth", 'rural water tanks charger',
              'rural water tanks discharger', 'rural water tanks']
    for c in ["Generator", "Store", "Link", "Line", "Bus"]:
        if "carrier" not in n.df(c).columns: continue
        assets_i = n.df(c)[n.df(c).carrier.isin(remove)].index
        if assets_i.empty: continue
        logger.info("removing {} of tech {}".format(c, assets_i))
        n.mremove(c, assets_i)

    # remove in the first investment periods
    remove = ["SMR CC", 'urban central solar thermal',
              'urban decentral solar thermal', 'rural solar thermal',
              'urban central solid biomass CHP CC',
              'urban central gas CHP CC', 'solid biomass for industry CC']
    first_years = n.snapshots.levels[0][n.snapshots.levels[0]<2040]
    for c in ["Generator", "Store", "Link", "Line"]:
        if "carrier" not in n.df(c).columns: continue
        assets_i = n.df(c)[(n.df(c).carrier.isin(remove)) &
                           (n.df(c).build_year.isin(first_years))].index
        if assets_i.empty: continue
        logger.info("removing {} of tech {}".format(c, assets_i))
        n.mremove(c, assets_i)


def add_conv_generators(n, carrier):
    conv_df = n.links[n.links.carrier==carrier]
    logger.info("add extendable conventionals for {}".format(carrier))
    for year in years:
        n.add("Link",
              "{}-{}".format(carrier, year),
              bus0=conv_df.bus0[0],
              bus1=conv_df.bus1[0],
              bus2=conv_df.bus2[0],
              p_nom_extendable=True,
              marginal_cost=conv_df.marginal_cost.mean(),
              capital_cost=conv_df.capital_cost.mean(),
              efficiency=conv_df.efficiency.mean(),
              efficiency2=conv_df.efficiency2.mean(),
              build_year=year,
              # carrier=carrier,
              lifetime=conv_df.lifetime.mean(),
              # location=conv_df.location[0]
              )
        n.links.loc["{}-{}".format(carrier, year), "carrier"] = carrier
        n.links.loc["{}-{}".format(carrier, year), "location"] = conv_df.location[0]
    return n


def island_hydrogen_production(n, export=False, compete=False):
    logger.info("Islanding hydrogen production")

    electrolysers = n.links.index[n.links.carrier == "H2 Electrolysis"]
    nodes = pd.Index(n.links.bus0[electrolysers].unique())

    n.madd("Bus",
           nodes + " electricity for hydrogen",
           location=nodes,
           carrier="electricity for hydrogen"
           )

    if export:
        print("Adding electricity export from hydrogen island")
        n.madd("Link",
               nodes + " export electricity from hydrogen island",
               bus0=nodes + " electricity for hydrogen",
               bus1=nodes,
               carrier="export electricity from hydrogen island",
               p_nom=1e6
               )

    if compete:
        n.links = pd.concat((n.links,n.links.loc[electrolysers].rename(lambda name: name + " for hydrogen")))
        n.links.loc[electrolysers + " for hydrogen","bus0"] += " electricity for hydrogen"
        # n.links.loc[electrolysers + " for hydrogen","carrier"] += " for hydrogen"
    else:
        n.links.loc[electrolysers,"bus0"] += " electricity for hydrogen"

    vre = n.generators.index[n.generators.carrier.isin(["onwind","solar","offwind-ac","offwind-dc", "offwind"]) & n.generators.p_nom_extendable]

    n.generators = pd.concat((n.generators,n.generators.loc[vre].rename(lambda name: name + " for hydrogen")))

    n.generators.loc[vre + " for hydrogen","bus"] += " electricity for hydrogen"
    # n.generators.loc[vre + " for hydrogen","carrier"] += " for hydrogen"

    n.generators_t.p_max_pu = pd.concat((n.generators_t.p_max_pu,n.generators_t.p_max_pu[vre].rename(lambda name: name + " for hydrogen",axis=1)),axis=1)
    #NB: Also have taken care of joint p_nom_max in extra_functionality between competing generation at each node

    # subtract upstream distribution grid costs added in add_electricity_grid_connection
    periods = n.investment_period_weightings.index
    cost_folder = snakemake.input.costs
    discount_rate = snakemake.config["costs"]["discountrate"]
    lifetime = snakemake.config["costs"]["lifetime"]
    costs = prepare_costs_all_years(
        periods, True, cost_folder, discount_rate, lifetime
    )
    onshore_hydrogen_vre = n.generators.index[n.generators.carrier.isin(["onwind","solar"]) & (n.generators.bus.str.contains("electricity for hydrogen"))]
    n.generators.loc[onshore_hydrogen_vre, "capital_cost"] -= costs[2020].at['electricity grid connection', 'fixed']
    n.generators.loc[onshore_hydrogen_vre, "nolearning_cost"] = 0.

    n.madd("Bus",
        nodes + " battery for hydrogen",
        location=nodes,
        carrier="battery"
    )

    for year in costs.keys():
        n.madd("Store",
            nodes + " battery for hydrogen-{}".format(year),
            bus=nodes + " battery for hydrogen",
            e_cyclic=True,
            e_nom_extendable=True,
            build_year=year,
            carrier="battery",
            capital_cost=costs[year].at['battery storage', 'fixed'],
            lifetime=costs[year].at['battery storage', 'lifetime']
        )

        n.madd("Link",
            nodes + " battery charger for hydrogen-{}".format(year),
            bus0=nodes + " electricity for hydrogen",
            bus1=nodes + " battery for hydrogen",
            carrier="battery charger",
            build_year=year,
            efficiency=costs[year].at['battery inverter', 'efficiency']**0.5,
            capital_cost=costs[year].at['battery inverter', 'fixed'],
            p_nom_extendable=True,
            lifetime=costs[year].at['battery inverter', 'lifetime']
        )

        n.madd("Link",
            nodes + " battery discharger for hydrogen-{}".format(year),
            bus0=nodes + " battery for hydrogen",
            bus1=nodes + " electricity for hydrogen",
            carrier="battery discharger",
            build_year=year,
            efficiency=costs[year].at['battery inverter', 'efficiency']**0.5,
            p_nom_extendable=True,
            lifetime=costs[year].at['battery inverter', 'lifetime']
        )


def cycling_shift(df, steps=1):
    """Cyclic shift on index of pd.Series|pd.DataFrame by number of steps"""
    df = df.copy()
    new_index = np.roll(df.index, steps)
    df.values[:] = df.reindex(index=new_index).values
    return df

def modify_transport_scenario(n, scenario):
    transport_scenario = pd.read_csv(snakemake.input.transport_scenarios,
                                     index_col=[0], header=[0,1])
    if scenario not in transport_scenario.columns.levels[0]:
        logger.warning("transport scenario not found in csv, keep default transport shares assumptions.\n")
        return
    # shares
    logger.info("set transport scenario {}".format(scenario))
    fuel_cell_share = transport_scenario[scenario]["land_transport_fuel_cell_share"]
    electric_share = transport_scenario[scenario]["land_transport_electric_share"]
    ice_share = 1 - fuel_cell_share - electric_share
    logger.info("Transport shares \n {}".format(transport_scenario[scenario].to_markdown()))

    # transport data
    transport = pd.read_csv(snakemake.input.transport, index_col=0, parse_dates=True)
    nodal_transport_data = pd.read_csv(snakemake.input.nodal_transport_data, index_col=0)

    # costs
    periods = n.investment_period_weightings.index
    cost_folder = snakemake.input.costs
    discount_rate = snakemake.config["costs"]["discountrate"]
    lifetime = snakemake.config["costs"]["lifetime"]
    costs = prepare_costs_all_years(
        periods, True, cost_folder, discount_rate, lifetime
    )

    nodes = n.buses[n.buses.carrier=="AC"].index

    # EV --------------------------
    # load
    periods = n.investment_period_weightings.index
    base_load = (transport[nodes] + cycling_shift(transport[nodes], 1) +
                 cycling_shift(transport[nodes], 2)) / 3
    load_ev = pd.concat([pd.concat([base_load.mul(electric_share[year])], keys=[year])
                         for year in periods])
    load_ev.columns = n.loads_t.p_set.loc[:, n.loads.carrier=="land transport EV"].columns
    n.loads_t.p_set.loc[:, n.loads.carrier=="land transport EV"] = load_ev

    # p nom
    p_nom = electric_share.apply(lambda x: x*nodal_transport_data["number cars"] *
                                 snakemake.config["sector"]["bev_charge_rate"])
    p_nom =  p_nom.rename(columns=lambda x: x + " low voltage").stack()
    # BEV charger
    wished_order = n.links[n.links.carrier=="BEV charger"].set_index(["build_year", "bus0"]).index
    p_nom_sort =  p_nom.reindex(wished_order)
    p_nom_sort.index = n.links[n.links.carrier=="BEV charger"].index
    n.links.loc[p_nom_sort.index, "p_nom"] = p_nom_sort
    # V2G
    if not n.links[n.links.carrier=="V2G"].index.empty:
        n.links.loc[p_nom_sort.index.str.replace("BEV charger", "V2G"), "p_nom"] = p_nom_sort.rename(lambda x: x.replace("BEV charger", "V2G"))
    # BEV DSM
    if not n.stores[n.stores.carrier=="battery storage"].index.empty:
        e_nom = electric_share.apply(lambda x: x*nodal_transport_data["number cars"]
                                      *  snakemake.config["sector"]["bev_energy"]
                                      *  snakemake.config["sector"]["bev_availability"])
        e_nom = e_nom.rename(columns=lambda x: x + " EV battery").stack()
        wished_order = n.stores[n.stores.carrier=="battery storage"].set_index(["build_year", "bus"]).index
        e_nom_sort =  e_nom.reindex(wished_order)
        e_nom_sort.index = n.stores[n.stores.carrier=="battery storage"].index
        n.stores.loc[e_nom_sort.index, "e_nom"] = e_nom_sort

    # FCEV ---------------------------------
    base_load = 1/snakemake.config["sector"]["transport_fuel_cell_efficiency"] * transport[nodes]
    load_fcev = pd.concat([pd.concat([base_load.mul(fuel_cell_share[year])], keys=[year])
                         for year in periods])
    load_fcev.columns = n.loads_t.p_set.loc[:, n.loads.carrier=="land transport fuel cell"].columns
    n.loads_t.p_set.loc[:, n.loads.carrier=="land transport fuel cell"] = load_fcev

    # ICE ----------------------------------
    ice_efficiency = snakemake.config["sector"]['transport_internal_combustion_efficiency']
    base_load = 1/ice_efficiency * transport[nodes]
    load_ice = pd.concat([pd.concat([base_load.mul(ice_share[year])], keys=[year])
                         for year in periods])
    # special case since load is grouped by bus and land transport oil all have the same bus
    load_ice = pd.DataFrame(load_ice.sum(axis=1), columns=n.loads_t.p_set.loc[:, n.loads.carrier=="land transport oil"].columns)
    n.loads_t.p_set.loc[:, n.loads.carrier=="land transport oil"] = load_ice
    # ICE emission
    co2 = ice_share / ice_efficiency * transport[nodes].sum().sum() / 8760 * costs[2020].at["oil", "CO2 intensity"]
    emissions = pd.DataFrame(-co2.reindex(n.snapshots, level=0), columns=n.loads[n.loads.carrier=="land transport oil emissions"].index)
    n.loads_t.p_set.loc[:,n.loads.carrier=="land transport oil emissions"] = emissions

    return n
#%%
if __name__ == "__main__":
    if "snakemake" not in globals():
        import os

        os.chdir("/home/lisa/Documents/learning_curve/scripts")
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "set_opts_and_solve",
            sector_opts="Co2L-1p5-notarget-5p24h-transportfast", #"-learnH2xElectrolysisp10-learnoffwindp10-learnonwindp10-learnsolarp10-seqcost",
            clusters="37",
        )

    years = snakemake.config["scenario"]["investment_periods"]
    # scenario options
    opts = snakemake.wildcards.sector_opts.split("-")

    n = pypsa.Network(
        snakemake.input.network, override_component_attrs=override_component_attrs
    )
    # TODO
    snakemake.config["co2_budget"]["1p5"] = 30

    # fix for current pypsa-eur-sec version
    co2_i = n.stores[n.stores.carrier=="co2 stored"].index
    n.stores.loc[co2_i, "e_nom_max"] = 200e6

    if any(["transport" in o for o in opts]):
        transport_scenario = [o for o in opts if "transport" in o][0]
        n = modify_transport_scenario(n, transport_scenario)
    if snakemake.config["cluster_heat_nodes"]:
        n = cluster_heat_buses(n)
    # cluster network spatially to one node
    if snakemake.config["one_node"]:
        n = cluster_network(n, years)
    # consider only some countries
    if len(snakemake.config["select_cts"]):
        n = select_cts(n, years)
    if snakemake.config["cluster_regions"]:
        n = cluster_to_regions(n)
    # must run condition for heating technologies
    if snakemake.config["heat_must_run"]:
        n = heat_must_run(n)


    # prepare data
    global_capacity, p_nom_max_limit, global_factor = prepare_data()
    # set scenario options
    n = set_scenario_opts(n, opts)

    # carbon emission constraints
    if "Co2L" in opts:
        n = set_carbon_constraints(n)

    # set max growth for renewables
    if snakemake.config["limit_growth"] or "limitgrowth" in opts:
        n = set_max_growth(n)

    # set min growth for renewables
    if snakemake.config["limit_growth_lb"]:
        n = set_min_growth(n)

    # TODO
    # extend lifetime of nuclear power plants to 60 years
    nuclear_i = n.links[n.links.carrier == "nuclear"].index
    n.links.loc[nuclear_i, "lifetime"] = 60.0

    # TODO DAC global capacity check
    if "DAC" in n.carriers.index:
        n.carriers.loc["DAC", "global_capacity"] = 10

    # aggregate network temporal
    if snakemake.config["temporal_presolve"] != "None":
        m = set_temporal_aggregation(n.copy(), [snakemake.config["temporal_presolve"]])
    n = set_temporal_aggregation(n, opts)

    # time delay learning
    time_delay = snakemake.config["time_delay"] and "notimedelay" not in opts
    logger.info("\n ------------\n Time delay is set to : {}\n ------------\n".format(time_delay))
    # solve network
    logging.basicConfig(
        filename=snakemake.log.python, level=snakemake.config["logging"]["level"]
    )
    # for solving the solution variables
    sols = pd.DataFrame()

    remove_techs_for_speed(n)

    if "nogridcost" in opts:
        logger.info("no grid connection costs for solar and onwind assumed")
        gen_i = n.generators[n.generators.carrier.isin(["solar", "onwind"])&n.generators.p_nom_extendable].index
        n.generators.loc[gen_i, "nolearning_cost"] = 0.

    # for carrier in ["nuclear", "lignite", "coal", "CCGT"]:
    #     n = add_conv_generators(n, carrier)

    # drop co2 vent
    logger.info("Remove co2 vent.")
    vent_i = n.links[n.links.carrier=="co2 vent"].index
    n.mremove("Link", vent_i)

    if "h2island" in opts:
        island_hydrogen_production(n, export=False, compete=False)

    #%%
    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=30.0
    ) as mem:

        if snakemake.config["temporal_presolve"] != "None":

            (
                extra_functionality_pre,
                skip_objective,
                typical_period,
                solver_options,
                keep_shadowprices,
                m,
            ) = prepare_network(m)

        (
            extra_functionality,
            skip_objective,
            typical_period,
            solver_options,
            keep_shadowprices,
            n,
        ) = prepare_network(n)
        # solver_options["threads"] = 8
        #

        if not "seqcost" in opts:
            # first run with low temporal resolution -----
            if snakemake.config["temporal_presolve"] != "None":

                logger.info(
                    "Solve network with lower temporal resolution \n"
                    "***********************************************************"
                )

                m.lopf(
                    pyomo=False,
                    solver_name="gurobi",
                    skip_objective=skip_objective,
                    multi_investment_periods=True,
                    solver_options=solver_options,
                    solver_logfile=snakemake.log.solver,
                    extra_functionality=extra_functionality_pre,
                    keep_shadowprices=keep_shadowprices,
                    typical_period=typical_period,
                    keep_references=True,
                    # warmstart=True
                    # store_basis=True,
                )
                n.start_fn = m.start_fn

                n.mapping = m.vars
                solver_options["NodeMethod"] = 2

            logger.info(
                "Solve network with {} snapshots \n"
                "***********************************************************".format(
                    len(n.snapshots)
                )
            )
            n.lopf(
                pyomo=False,
                solver_name="gurobi",
                skip_objective=skip_objective,
                multi_investment_periods=True,
                solver_options=solver_options,
                solver_logfile=snakemake.log.solver,
                extra_functionality=extra_functionality,
                keep_shadowprices=keep_shadowprices,
                typical_period=typical_period,
                keep_references=True,
                # warmstart=True,
                # store_basis=True,
            )
            # for debugging
            if hasattr(n, "objective") and hasattr(n.sols, "Carrier"):
                try:
                    for key in n.sols["Carrier"]["pnl"].keys():
                        sol_attr = round(n.sols["Carrier"]["pnl"][key].groupby(level=0).first(), ndigits=2)
                        if not isinstance(sol_attr.columns, pd.MultiIndex):
                            segments_i = pd.Index(np.arange(snakemake.config["segments"]+1))
                            sol_attr = sol_attr.reindex(
                                pd.MultiIndex.from_product([sol_attr.columns, segments_i]),
                                level=0,
                                axis=1,
                            )
                        sol_attr = pd.concat([sol_attr], keys=[key], axis=1)
                        sols = pd.concat([sols, sol_attr], axis=1)
                except AttributeError:
                    logger.info("no sols saved")
        # solve linear sequential problem with cost update for technology learning
        else:
            seqlopf(
                n,
                min_iterations=4,
                max_iterations=10,
                track_iterations=True,
                msq_threshold=0.05,
                extra_functionality=extra_functionality,
                time_delay=time_delay,
            )
            n.buses_t.v_ang = n.buses_t.v_ang.astype(float)
#%%
        n.export_to_netcdf(snakemake.output.network)

        sols.to_csv(snakemake.output.sols)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
