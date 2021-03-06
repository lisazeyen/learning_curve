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

#%% FUNCTIONS ---------------------------------------------------------------
aggregate_dict = {
    "p_nom": "sum",
    "p_nom_max": "sum",
    "p_nom_min": "sum",
    "p_nom_max": "sum",
    "p_set": "sum",
    "e_initial": "sum",
    "e_nom": "sum",
    "e_nom_max": "sum",
    "e_nom_min": "sum",
    "state_of_charge_initial": "sum",
    "state_of_charge_set": "sum",
    "inflow": "sum",
    "p_max_pu": "mean",
}


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
                    n.carriers.loc[tech, "initial_cost"] = (
                        n.df(c).loc[index, "capital_cost"].mean()
                    )
                # TODO
                if tech == "H2 electrolysis":
                    n.carriers.loc["H2 electrolysis", "max_capacity"] = 1.2e6 / factor
                if tech == "H2 Electrolysis":
                    n.carriers.loc["H2 Electrolysis", "max_capacity"] = 1.2e6 / factor
                if tech == "H2 Fuel Cell":
                    n.carriers.loc["H2 Fuel Cell", "max_capacity"] = 2e4
                if tech == "DAC":
                    n.carriers.loc["DAC", "max_capacity"] = 120e3 / factor
                if tech == "solar":
                    n.carriers.loc["solar", "max_capacity"] = 4e6 / factor
                if tech == "onwind":
                    n.carriers.loc["onwind", "max_capacity"] = 4e6 / factor
        if "fcev" in o:
            fcev_fraction = float(o.replace("fcev", "")) / 100
            n = adjust_land_transport_share(n, fcev_fraction)

        if "co2seq" in o:
            factor = float(o.replace("co2seq", ""))
            n.stores.loc["co2 stored-2020", "e_nom_max"] *= factor
            logger.info(
                "Total CO2 sequestration potential is set to {}".format(
                    n.stores.loc["co2 stored-2020", "e_nom_max"]
                )
            )
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
        snakemake.config["co2_budget"]["1p5"] * 1e9
    )  # budget for + 1.5 Celsius for Europe
    logger.info("add carbon budget of {}".format(budget))
    n.add(
        "GlobalConstraint",
        "Budget",
        type="Budget",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=budget,
    )

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
            con = write_constraint(n, lhs, "==", rhs, axes=pd.Index([name]))
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
    )

    define_constraints(n, lhs, "<=", rhs, "GlobalConstraint", "res_limit")


def add_capacity_constraint(n, snapshots):
    """Additional constraint for overall installed capacity to speed up optimisation."""
    c, attr = "Generator", "p_nom"
    res = ["offwind-ac", "offwind-dc", "onwind", "solar", "solar rooftop"]
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

    rhs = pd.Series([140e3, 200e3, 900e3, 700e3, 400e3], index=res)

    define_constraints(n, lhs, ">=", rhs, "GlobalConstraint", "min_cap")


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

    lhs = linexpr((-1, h2_p), (1, gas_boiler_cap_t.values), (-1, gas_boiler_p.values))
    rhs = 0.0
    define_constraints(n, lhs, ">=", rhs, "Link", "retro_gasboiler")


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
                n, snapshots, segments=snakemake.config["segments"], time_delay=True
            )
            add_carbon_neutral_constraint(n, snapshots)
            add_local_res_constraint(n, snapshots)
            if snakemake.config["h2boiler_retrofit"]:
                add_retrofit_gas_boilers_constraint(n, snapshots)
            # add_capacity_constraint(n, snapshots)

        skip_objective = True
        keep_shadowprices = False
    else:

        def extra_functionality(n, snapshots):
            add_battery_constraints(n)
            add_carbon_neutral_constraint(n, snapshots)
            add_local_res_constraint(n, snapshots)
            if snakemake.config["h2boiler_retrofit"]:
                add_retrofit_gas_boilers_constraint(n, snapshots)
            # add_capacity_constraint(n, snapshots)

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

        for t in n.iterate_components(["Line", "Link"]):
            np.random.seed(123)
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

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

    def get_new_cost(n, c):
        """Calculate new cost depending on installed capacities."""
        assets = n.df(c).loc[prev[c].index]
        learn_carriers = assets.carrier.unique()

        # learning rate
        learning_rate = n.carriers.loc[learn_carriers, "learning_rate"]
        # initial installed capacity
        initial_cum_cap = n.carriers.loc[learn_carriers, "global_capacity"]
        # initial cost
        c0 = n.carriers.loc[learn_carriers, "initial_cost"]

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

        # (a) investment costs by piecewise linearisation
        x_high = n.carriers.loc[learn_carriers, "max_capacity"]
        segments = snakemake.config["segments"]
        points = get_linear_interpolation_points(n, initial_cum_cap, x_high, segments)
        slope = get_slope(points)

        cost = pd.DataFrame(columns=learn_carriers)
        for carrier in learn_carriers:
            cost[carrier] = (
                cum_p_nom.loc[carrier]
                .apply(
                    lambda x: points.xs("x_fit", level=1, axis=1)[carrier][
                        (x >= points.xs("x_fit", level=1, axis=1)[carrier])
                    ].index[-1]
                )
                .map(slope[carrier])
            )

        # (b) investment costs exactly form learning curve
        capital_cost = cum_p_nom.apply(
            lambda x: experience_curve(
                x,
                learning_rate.loc[x.name],
                c0.loc[x.name],
                initial_cum_cap.loc[x.name],
            ),
            axis=1,
        )

        return pd.Series(
            n.df(c).set_index(["carrier", "build_year"]).index.map(cost.T.stack()),
            index=n.df(c).index,
        ).fillna(n.df(c).capital_cost)

    def update_cost_params(n, prev):
        """Update cost parameters according to installed capacities."""
        for c in prev.keys():

            # overwrite cost assumptions
            n.df(c)["capital_cost"] = get_new_cost(n, c)

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
            n.df(c)[f"capital_cost_{iteration}"] = n.df(c)["capital_cost"]
        setattr(n, f"status_{iteration}", status)
        setattr(n, f"objective_{iteration}", n.objective)
        n.iteration = iteration
        # n.global_constraints = n.global_constraints.rename(
        #     columns={"mu": f"mu_{iteration}"}
        # )
        n.global_constraints[f"mu_{iteration}"] = n.global_constraints["mu"]

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
        update_cost_params(n, prev)
        # calculate difference between previous and current result
        diff = msq_diff(n, prev)
        del n.start_fn
        iteration += 1


#%%
if __name__ == "__main__":
    if "snakemake" not in globals():
        import os

        os.chdir("/home/lisa/Documents/learning_curve/scripts")
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "set_opts_and_solve",
            sector_opts="Co2L-73sn-learnH2xElectrolysisp0-learnH2xFuelxCellp0-learnDACp0-learnsolarp0-learnonwindp0-co2seq1-local",
            clusters="37",
        )

    years = snakemake.config["scenario"]["investment_periods"]
    # scenario options
    opts = snakemake.wildcards.sector_opts.split("-")

    n = pypsa.Network(
        snakemake.input.network, override_component_attrs=override_component_attrs
    )

    # cluster network spatially to one node
    if snakemake.config["one_node"]:
        n = cluster_network(n, years)
    # consider only some countries
    if len(snakemake.config["select_cts"]):
        n = select_cts(n, years)
    # prepare data
    global_capacity, p_nom_max_limit, global_factor = prepare_data()
    # set scenario options
    n = set_scenario_opts(n, opts)

    # carbon emission constraints
    if "Co2L" in opts:
        n = set_carbon_constraints(n)

    # set max growth for renewables
    if snakemake.config["limit_growth"]:
        n = set_max_growth(n)

    # set min growth for renewables
    if snakemake.config["limit_growth_lb"]:
        n = set_min_growth(n)

    # TODO
    # extend lifetime of nuclear power plants to 60 years
    nuclear_i = n.links[n.links.carrier == "nuclear"].index
    n.links.loc[nuclear_i, "lifetime"] = 60.0
    # TODO
    bev_dsm_i = n.stores[n.stores.carrier == "battery storage"].index
    n.stores.loc[bev_dsm_i, "e_nom_max"] = n.stores.loc[bev_dsm_i, "e_nom"]
    n.stores.loc[bev_dsm_i, "e_nom_extendable"] = True
    n.stores.loc[bev_dsm_i, "carrier"] = "battery"

    # aggregate network temporal
    if snakemake.config["temporal_presolve"] != "None":
        m = set_temporal_aggregation(n.copy(), [snakemake.config["temporal_presolve"]])
    n = set_temporal_aggregation(n, opts)

    # solve network
    logging.basicConfig(
        filename=snakemake.log.python, level=snakemake.config["logging"]["level"]
    )

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
        #%%

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
        # solve linear sequential problem with cost update for technology learning
        else:
            seqlopf(
                n,
                min_iterations=4,
                max_iterations=6,
                track_iterations=True,
                msq_threshold=0.05,
                extra_functionality=extra_functionality,
            )

        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
