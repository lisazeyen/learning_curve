#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:53:04 2021
Prepare sector coupling network for perfect foresight optimisation

@author: bw0928
"""
import pypsa_learning as pypsa
import pandas as pd
import numpy as np
import logging

from pypsa_learning.io import import_components_from_dataframe
from six import iterkeys
from pypsa_learning.descriptors import expand_series
from vresutils.costdata import annuity
import re

from distutils.version import LooseVersion
import xarray as xr

pd_version = LooseVersion(pd.__version__)
agg_group_kwargs = dict(numeric_only=False) if pd_version >= "1.3" else {}


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

override_old = override_component_attrs.copy()
override_old["Link"].loc["build_year"] = [
    "integer",
    "year",
    np.nan,
    "build year",
    "Input (optional)",
]
override_old["Link"].loc["lifetime"] = [
    "float",
    "years",
    np.nan,
    "build year",
    "Input (optional)",
]
override_old["Generator"].loc["build_year"] = [
    "integer",
    "year",
    np.nan,
    "build year",
    "Input (optional)",
]
override_old["Generator"].loc["lifetime"] = [
    "float",
    "years",
    np.nan,
    "build year",
    "Input (optional)",
]
override_old["Store"].loc["build_year"] = [
    "integer",
    "year",
    np.nan,
    "build year",
    "Input (optional)",
]
override_old["Store"].loc["lifetime"] = [
    "float",
    "years",
    np.nan,
    "build year",
    "Input (optional)",
]

#%%  MAPPING
# maps pypsa name to technology data name
pypsa_to_techbase = {
    "H2 electrolysis": "electrolysis",
    "H2 Electrolysis": "electrolysis",
    "H2 Fuel Cell": "fuel cell",
    "H2 fuel cell": "fuel cell",
    "battery charger": "battery inverter",
    # 'battery discharger':"battery inverter",
    "H2": "hydrogen storage underground",
    "battery": "battery storage",
    "offwind-ac": "offwind",
    "offwind-dc": "offwind",
    "solar rooftop": "solar-rooftop",
    "solar": "solar-utility",
    "DAC": "direct air capture",
}

heat_carriers = {
    "residential rural": "decentral",
    "services rural": "decentral",
    "urban central": "central",
    "residential urban decentral": "decentral",
    "services urban decentral": "decentral",
}

# FUNCTIONS ----------------------------------------------------------------
def concat_networks(years, with_time=True, snapshots=None, investment_periods=None):
    """Concat given pypsa networks and adds build_year.

        Returns
        --------
        n : pypsa.Network for the whole planning horizon

        Parameters
        ----------
        with_time : boolean, default True
            Copy snapshots and time-varying network.component_names_t data too.
        snapshots : list or index slice
            A list of snapshots to copy, must be a subset of
            network.snapshots, defaults to network.snapshots

        Examples
        --------
        >>> network_copy = network.copy()

        """
    # input paths of sector coupling networks
    network_paths = [snakemake.input.brownfield_network] + snakemake.input.network[1:]
    # final concatenated network
    n = pypsa.Network(override_component_attrs=override_component_attrs)

    pattern = re.compile(r"-\d\d\d\d")

    # helper functions ---------------------------------------------------
    def rename_df(df, new_i, axis=0):
        return df.rename(lambda x: x + "-" + str(year) if x in new_i else x, axis=axis)

    def get_already_build(df, pattern=re.compile(r"-\d\d\d\d")):
        # assets which are build earlier
        early_build_i = df[
            df.rename(lambda x: True if pattern.search(x) != None else False).index
        ].index
        # assets which will be newly build
        new_i = df.index.difference(early_build_i)

        return early_build_i, new_i

    def get_missing(df, n, c):
        """Get in network n missing assets of df for component c.

            Input:
                df: pandas DataFrame, static values of pypsa components
                n : pypsa Network to which new assets should be added
                c : string, pypsa component.list_name (e.g. "generators")
            Return:
                pd.DataFrame with static values of missing assets
            """
        df_final = getattr(n, c)
        missing_i = df.index.difference(df_final.index)
        return df.loc[missing_i]

    # --------------------------------------------------------------------

    # iterate over single year networks and concat to perfect foresight network
    for i, network_path in enumerate(network_paths):
        year = years[i]
        network = pypsa.Network(network_path, override_component_attrs=override_old)
        network.lines["carrier"] = "AC"

        # TODO can be removed once lifetime and build_year defaults are aligned with new pypsa
        for c in ["Generator", "Store", "Link", "Line"]:
            network.df(c)["build_year"].fillna(year, inplace=True)
            network.df(c)["lifetime"].fillna(np.inf, inplace=True)

        # static ----------------------------------
        # (1) add buses and carriers
        for component in network.iterate_components(["Bus", "Carrier"]):
            df_year = component.df
            # get missing assets
            missing = get_missing(df_year, n, component.list_name)
            import_components_from_dataframe(n, missing, component.name)
        # (2) add generators, links, stores and loads
        for component in network.iterate_components(
            ["Generator", "Link", "Store", "Load", "Line"]
        ):

            df_year = component.df.copy()
            # assets which are build earlier
            early_build_i, new_i = get_already_build(df_year)
            # set build year of assets
            df_year["build_year"] = df_year.rename(
                lambda x: int(pattern.search(x).group().replace("-", ""))
                if x in early_build_i
                else year
            ).index
            df_year = rename_df(df_year, new_i, axis=0)

            missing = get_missing(df_year, n, component.list_name)

            import_components_from_dataframe(n, missing, component.name)

        # time variant --------------------------------------------------
        if with_time:
            if snapshots is None:
                snapshots = network.snapshots
            n.set_snapshots(snapshots)
            investment_periods = network.investment_period_weightings.index
            for component in network.iterate_components():
                pnl = getattr(n, component.list_name + "_t")
                for k in iterkeys(component.pnl):
                    if component.name in ["Generator", "Link", "Store", "Load", "Line"]:
                        df_year = component.df
                        # assets which are build earlier
                        early_build_i, new_i = get_already_build(df_year)
                        pnl_year = component.pnl[k].loc[snapshots].copy()
                        pnl_year = rename_df(pnl_year, new_i, axis=1)

                        if component.name == "Load" and k == "p_set":
                            static_load = network.loads.loc[network.loads.p_set != 0]
                            static_load = rename_df(static_load, new_i, axis=0)
                            static_load_t = expand_series(
                                static_load.p_set, network.snapshots
                            ).T
                            pnl_year = pd.concat([pnl_year, static_load_t], axis=1)
                        pnl[k] = pd.concat([pnl[k], pnl_year], axis=1)
                    elif component.name in ["Bus", "Carrier"]:
                        df_year = component.df
                        missing = get_missing(df_year, n, component.list_name)
                        cols = missing.index.intersection(component.pnl[k].columns)
                        pnl_year = component.pnl[k].loc[snapshots].copy()
                        pnl[k] = pd.concat([pnl[k], pnl_year[cols]], axis=1)
                    else:
                        pnl[k] = component.pnl[k]

            n.snapshot_weightings = network.snapshot_weightings.loc[snapshots].copy()
            n.investment_period_weightings = network.investment_period_weightings.loc[
                investment_periods
            ].copy()
        else:
            investment_periods = network.investment_period_weightings.index
            n.snapshot_weightings = network.snapshot_weightings.copy()
            n.investment_period_weightings = network.investment_period_weightings.loc[
                investment_periods
            ].copy()

    n.loads["p_set"] = 0
    # drop old global constraints
    n.global_constraints.drop(n.global_constraints.index, inplace=True)

    return n


def prepare_costs(cost_file, discount_rate, lifetime):
    """Prepare cost data."""
    # set all asset costs and other parameters
    costs = pd.read_csv(cost_file, index_col=list(range(2))).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3

    # min_count=1 is important to generate NaNs which are then filled by fillna
    costs = (
        costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    )
    costs = costs.fillna(
        {
            "CO2 intensity": 0,
            "FOM": 0,
            "VOM": 0,
            "discount rate": discount_rate,
            "efficiency": 1,
            "fuel": 0,
            "investment": 0,
            "lifetime": lifetime,
        }
    )

    annuity_factor = lambda v: annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100
    costs["fixed"] = [annuity_factor(v) * v["investment"] for i, v in costs.iterrows()]

    return costs


def prepare_costs_all_years(
    years, update=True, cost_folder="data/costs/", discountrate=0.07, lifetime=25
):
    """Prepare cost data for multiple years."""
    all_costs = {}

    for year in years:
        all_costs[year] = prepare_costs(
            cost_folder + "/costs_{}.csv".format(year), discountrate, lifetime,
        )

    if not update:
        for year in years:
            all_costs[year] = all_costs[years[0]]

    return all_costs


def update_costs(n, costs_dict, years, update):
    """Update all costs according to costs in the first investment period."""


    costs = pd.concat(costs_dict)

    def relabel_pypsa_to_techbase(label):
        """Rename pypsa tech names to names in database."""
        for pypsa_heat_c, tech_heat_c in heat_carriers.items():
            label = label.replace(pypsa_heat_c, tech_heat_c)
        if label in pypsa_to_techbase.keys():
            label = pypsa_to_techbase[label]
        return label

    def get_new_attr(df, attr, costs):
        filler = df[attr]
        attr = attr.replace("capital_cost", "fixed")
        cost_index = (
            df.set_index(["build_year", "carrier"])
            .rename(relabel_pypsa_to_techbase, level=1)
            .index
        )
        filler.index = cost_index
        return costs.reindex(cost_index)[attr].fillna(filler)

    def get_first_attr(df, attr):
        map_dict = (
            df[attr]
            .groupby([df.carrier, df.build_year])
            .first(**agg_group_kwargs)
            .sort_index()
            .groupby(level=0)
            .first()
        )
        return df.carrier.map(map_dict)

    for c in n.iterate_components(n.one_port_components | n.branch_components):
        df = c.df
        if df.empty or (c.name in (["Line", "StorageUnit", "Load"])):
            continue

        # TODO very error prune!! rather rerun pypsa-eur-sec
        # # update lifetime + investment costs
        # for attr in ["capital_cost", "lifetime"]:
        #     if attr not in c.df.columns:
        #         continue
        #     df[attr] = get_new_attr(df, attr, costs).values

        if not update:
            logger.info(
                "\n***********************************************************\n"
                "Set capital-,marginal cost and efficiencies"
                "on first investment period assumptions. \n"
                "***********************************************************\n"
            )
            # update costs and efficiencies
            attrs = [
                "capital_cost",
                "marginal_cost",
                "efficiency",
                "efficiency2",
                "efficiency3",
                "efficiency4",
            ]
            for attr in attrs:
                if attr not in c.df.columns:
                    continue
                c.df[attr] = get_first_attr(c.df, attr)

    # update offshore wind costs with connection costs
    update_offwind_costs(n, costs, years)

    # extract electricity grid connection costs for onwind and solar
    # CAREFUL: inconsistency with brownfield since to electricity connection costs are added
    if snakemake.config["sector"]['electricity_grid_connection']:
        logger.info("\n -----------------------\n"
                    "Electricity grid connection cost for onwind, solar and offwind "
                    "are considered not to experience any learning."
                    "\n ----------------------\n")
        gens_i = n.generators[n.generators.p_nom_extendable &
                              n.generators.carrier.isin(["onwind", "solar"])].index
        grid_con_cost = costs.xs('electricity grid connection', level=1)["fixed"]
        n.generators.loc[gens_i, "nolearning_cost"] = n.generators.loc[gens_i, "build_year"].map(grid_con_cost)


def update_offwind_costs(n, costs, years):
    """
    Update costs for offshore-wind generators added with pypsa-eur to those
    cost in the planning year
    """
    logger.info(
        "\n***********************************************************\n"
        "Add non-learning investment costs of offshore wind farms.\n"
        "***********************************************************\n"
    )
    # assign clustered bus
    # map initial network -> simplified network
    busmap_s = pd.read_csv(snakemake.input.busmap_s, index_col=0).squeeze()
    busmap_s.index = busmap_s.index.astype(str)
    busmap_s = busmap_s.astype(str)
    # map simplified network -> clustered network
    busmap = pd.read_csv(snakemake.input.busmap, index_col=0).squeeze()
    busmap.index = busmap.index.astype(str)
    busmap = busmap.astype(str)
    # map initial network -> clustered network
    clustermaps = busmap_s.map(busmap)

    # code adapted from pypsa-eur/scripts/add_electricity.py
    for connection in ["dc", "ac"]:
        tech = "offwind-" + connection
        profile = snakemake.input["profile_offwind_" + connection]
        # costs for different investment periods
        cost_submarine = costs.xs(tech + "-connection-submarine", level=1)["fixed"]
        cost_underground = costs.xs(tech + "-connection-underground", level=1)["fixed"]

        with xr.open_dataset(profile) as ds:
            underwater_fraction = ds["underwater_fraction"].to_pandas()
            # expand series for underwater fraction
            underwater_fraction_t = expand_series(underwater_fraction, years)
            # average distance
            av_dis_t = expand_series(ds["average_distance"].to_pandas(), years)
            # connection costs
            connection_cost = (
                snakemake.config["costs"]["lines"]["length_factor"]
                * av_dis_t
                * (
                    underwater_fraction_t.mul(cost_submarine)
                    + (1.0 - underwater_fraction_t).mul(cost_underground)
                )
            )

            # convert to aggregated clusters with weighting
            weight = ds["weight"].to_pandas()

            # e.g. clusters == 37m means that VRE generators are left
            # at clustering of simplified network, but that they are
            # connected to 37-node network
            # if snakemake.wildcards.clusters[-1:] == "m":
            #     genmap = busmap_s
            # else:
            genmap = clustermaps

            connection_cost = (
                connection_cost.mul(weight, axis=0)
                .groupby(genmap)
                .sum()
                .div(weight.groupby(genmap).sum(), axis=0)
            )
            # add station costs where no learning is assumed
            no_learn_cost = connection_cost.add(
                costs.xs("offwind-ac-station", level=1)["fixed"]
            )

            gen_b = (n.generators.carrier == tech) & (n.generators.p_nom_extendable)
            gen_i = n.generators[gen_b].set_index(["bus", "build_year"]).index
            n.generators.loc[gen_b, "nolearning_cost"] = (
                no_learn_cost.stack().reindex(gen_i).values
            )
            # costs with technology learning
            learn_costs = costs.xs("offwind", level=1)["fixed"]
            learn_costs_gen = n.generators[gen_b].build_year.map(learn_costs)

            capital_cost = learn_costs_gen + n.generators.loc[gen_b, "nolearning_cost"]

            n.generators.loc[
                gen_b, "capital_cost"
            ] = capital_cost  # .rename(index=lambda node: node + " " + tech)


def update_capacities(n):
    """Update already installed capacities of powerplants.

    The update is based on IRENA data [1] for renewables (including nuclear)
    and beyond-coal [2] for coal. This data is only country resolved. The
    capacities are updated because of larger RES capacities and lower conventional
    capacities compared to the power plant matching data.
    Capacities are distributed within a country weighted by capacities from
    power plant matching.


    [1] IRENA report
    Renewable Capacity Statistics (2020)
    https://irena.org/publications/2020/Mar/Renewable-Capacity-Statistics-2020

    [2] beyond coal
    https://beyond-coal.eu/wp-content/uploads/2021/05/2021-04-20_Europe_Beyond_Coal-European_Coal_Database_hcnn.xlsx
    sheet country (April 2021)
    """
    # considered countries
    countries = n.buses.country.unique()
    # add country to low voltage level
    bus_low_i = n.buses[n.buses.carrier == "low voltage"].index
    n.buses.loc[bus_low_i, "country"] = n.buses.loc[bus_low_i].index.str[:2]
    n.generators["country"] = n.generators.bus.map(n.buses.country)

    res = ["nuclear", "offwind-dc", "onwind", "solar"]
    res_i = n.generators[n.generators.carrier.isin(res)].index
    # weight for share of already existing capacity
    weight = (
        n.generators.p_nom
        / n.generators.p_nom.groupby(
            [n.generators.carrier, n.generators.country]
        ).transform("sum", **agg_group_kwargs)
    ).fillna(0)

    # installed renewable capacities according to IRENA
    local_capacity = pd.read_csv(snakemake.input.local_capacity, index_col=0)
    local_capacity = local_capacity[local_capacity.alpha_2.isin(countries)]
    local_caps = local_capacity.rename(index={"offwind": "offwind-dc"}).set_index(
        "alpha_2", append=True
    )
    local_caps = local_caps.groupby(level=[0, 1]).sum()

    p_nom = local_caps.reindex(
        n.generators.set_index(["carrier", "country"]).index
    ).set_index(n.generators.index)["Electricity Installed Capacity (MW)"]
    n.generators.loc[res_i, "p_nom"] = p_nom.mul(weight).fillna(0).loc[res_i]

    # Links -------------------------------------------------
    # nuclear
    links_i = n.links.loc[n.links.carrier == "nuclear"].index
    n.links.loc[links_i, "country"] = n.links.bus1.map(n.buses.country)
    p_nom = local_caps.reindex(
        n.links.set_index(["carrier", "country"]).index
    ).set_index(n.links.index)["Electricity Installed Capacity (MW)"]

    weight = (
        n.links.p_nom
        / n.links.p_nom.groupby([n.links.carrier, n.links.country]).transform(
            "sum", **agg_group_kwargs
        )
    ).fillna(0)
    n.links.loc[links_i, "p_nom"] = (p_nom.mul(weight) / n.links.efficiency).loc[
        links_i
    ]

    # coal
    # conventional capacities from other sources -> TODO move to script of local capacities
    p_nom.loc["lignite"] = 49760
    p_nom.loc["coal"] = 76644  # hard coal

    weight = (
        n.links.p_nom
        / n.links.p_nom.groupby(n.links.carrier).transform("sum", **agg_group_kwargs)
    ).fillna(0)

    conventional = ["lignite", "coal"]
    links_i = n.links.loc[n.links.carrier.isin(conventional)].index
    n.links.loc[links_i, "p_nom"] = (
        n.links.carrier.map(p_nom) * weight / n.links.efficiency
    ).loc[links_i]
    n.links.loc[links_i, "p_nom_extendable"] = False


def get_social_discount(t, r=0.01):
    """Calculate for a given time t the social discount."""
    return 1 / (1 + r) ** t


def get_investment_weighting(energy_weighting, r=0.01):
    """Define cost weighting.

    Returns cost weightings depending on the the energy_weighting (pd.Series)
    and the social discountrate r
    """
    end = energy_weighting.cumsum()
    start = energy_weighting.cumsum().shift().fillna(0)
    return pd.concat([start, end], axis=1).apply(
        lambda x: sum([get_social_discount(t, r) for t in range(int(x[0]), int(x[1]))]),
        axis=1,
    )


def set_multi_index(n, years, social_discountrate):
    """Set snapshots to pd.MultiImdex."""
    loads_t = (
        n.loads_t.p_set.groupby(
            [n.loads.carrier, n.loads.bus, n.loads.build_year], axis=1
        )
        .sum(**agg_group_kwargs)
        .stack()
        .swaplevel()
        .sort_index()
        .fillna(0)
    )
    helper = n.loads.reset_index()
    load_df_i = helper.groupby([helper.carrier, helper.bus]).first(**agg_group_kwargs)[
        "name"
    ]
    loads_t.columns = loads_t.columns.map(load_df_i)
    # loads_t.rename(columns=lambda x: x.replace("-2020", ""), inplace=True)
    loads_t.index.rename(["investment_period", "snapshot"], inplace=True)
    load_df = n.loads.loc[load_df_i]
    load_df.rename(index=lambda x: x.replace("-{}".format(years[0]), ""), inplace=True)
    loads_t.rename(columns=lambda x: x.replace("-{}".format(years[0]), ""), inplace=True)

    n.mremove("Load", n.loads.index)

    snapshots = pd.MultiIndex.from_product([years, n.snapshots])
    n.set_snapshots(snapshots)

    import_components_from_dataframe(n, load_df, "Load")
    n.loads_t.p_set = loads_t
    # set investment_weighting
    n.investment_period_weightings.loc[:, "time_weightings"] = (
        n.investment_period_weightings.index.to_series()
        .diff()
        .shift(-1)
        .values
    )
    n.investment_period_weightings.loc[:, "time_weightings"].fillna(method="ffill", inplace=True)
    n.investment_period_weightings.loc[
        :, "objective_weightings"
    ] = get_investment_weighting(
        n.investment_period_weightings["time_weightings"], social_discountrate
    )


def set_fixed_assets(c, fixed_carrier, overwrite_lifetime=True):
    """Reduce number of assets for a given component c and adjust lifetime.

    Input:
        c                 : pypsa.Component (e.g. "Generator")
        fixed_carrier     : list of carriers (e.g. ["coal", "lignite"])
        overwrite_lifetime: Bool. If True. lifetimes of first year are set to
                            infinity

    """
    fix = n.df(c)[n.df(c).carrier.isin(fixed_carrier)]
    build_first = fix[fix.build_year <= years[0]]

    to_drop = fix.index.difference(build_first.index)
    for asset in to_drop:
        n.remove(c, asset)

    if overwrite_lifetime:
        build_in_year = build_first[build_first.build_year == years[0]].index
        n.df(c).loc[build_in_year, "lifetime"] = np.inf


def set_assets_without_multiinvestment():
    """Reduce number of assets which do not neeed multiinvestment."""
    # generators ----------------
    conventionals = ["coal", "gas", "lignite", "oil", "ror", "uranium"]
    set_fixed_assets("Generator", conventionals)

    # links ---------------------
    links_fixed = [
        "CCGT",
        "biogas to gas",
        "co2 vent",
        "coal",
        # "electricity distribution grid",
        "gas for industry",
        "gas for industry CC",
        "lignite",
        "DC",
        "nuclear",
        "oil",
        "process emissions",
        # "battery discharger",
        # "home battery discharger",
        "process emissions CC",
        "residential rural water tanks charger",
        "residential rural water tanks discharger",
        "residential urban decentral water tanks charger",
        "residential urban decentral water tanks discharger",
        "services rural water tanks charger",
        "services rural water tanks discharger",
        "services urban decentral water tanks charger",
        "services urban decentral water tanks discharger",
        "solid biomass for industry",
        "solid biomass for industry CC",
        "urban central water tanks charger",
        "urban central water tanks discharger",
    ]
    set_fixed_assets("Link", links_fixed)

    # lines --------------------------------------------------------
    set_fixed_assets("Line", ["AC"])

    # stores --------------------------------------------------------
    stores_fixed = [
        "co2",
        "co2 stored",
        "coal",
        "gas",
        "lignite",
        "oil",
        "uranium",
        # "H2 Store",
    ]
    set_fixed_assets("Store", stores_fixed)


# %%
if __name__ == "__main__":
    if "snakemake" not in globals():
        import os

        os.chdir("/home/lisa/Documents/learning_curve/scripts")
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_perfect_foresight",
            sector_opts="146sn-learnH2xelectrolysisp0-co2seq1",
            clusters="37",
        )

    # parameters -----------------------------------------------------------
    years = snakemake.config["scenario"]["investment_periods"]
    social_discountrate = snakemake.config["costs"]["social_discountrate"]

    # concat prenetworks of planning horizon to single network ------------
    n = concat_networks(years)

    # costs ----------------------------------------------------------------
    update = snakemake.config["costs"]["update_costs"]
    cost_folder = snakemake.input.costs
    discount_rate = snakemake.config["costs"]["discountrate"]
    lifetime = snakemake.config["costs"]["lifetime"]
    costs = prepare_costs_all_years(years, update, cost_folder, discount_rate, lifetime)
    update_costs(n, costs, years, update)

    # adjust already installed capacities to latest IRENA report -----------
    update_capacities(n)

    # set snapshots MultiIndex and investment weightings -------------------
    set_multi_index(n, years, social_discountrate)

    # set assets which only need single investment -------------------------
    set_assets_without_multiinvestment()

    # co2 store assumptions
    co2_i = n.stores[n.stores.carrier.isin(["co2", "co2 stored"])].index
    n.stores.loc[co2_i, "e_period"] = False
    # increase biomass potentials
    store_i = n.stores[n.stores.carrier.isin(["biogas", "solid biomass"])].index
    time_weightings = n.stores.loc[store_i, "build_year"].map(
        n.investment_period_weightings["time_weightings"]
    )
    # n.stores.loc[store_i, ["e_nom", "e_initial"]] = n.stores.loc[store_i, ["e_nom", "e_initial"]].mul(time_weightings, axis=0)
    n.stores.loc[store_i, "lifetime"] = time_weightings

    # adjust lifetime of BEV and V2G
    to_adjust = ["V2G", "BEV charger"]
    to_adjust_i = n.links[n.links.carrier.isin(to_adjust)].index
    time_weightings = n.links.loc[to_adjust_i, "build_year"].map(
        n.investment_period_weightings["time_weightings"]
    )
    n.links.loc[to_adjust_i, "lifetime"] =  time_weightings
    to_adjust = ["Li ion", "battery storage"]
    to_adjust_i = n.stores[n.stores.carrier.isin(to_adjust)].index
    time_weightings = n.stores.loc[to_adjust_i, "build_year"].map(
        n.investment_period_weightings["time_weightings"]
    )
    n.stores.loc[to_adjust_i, "lifetime"] =  time_weightings

    # add hydrogen boilers
    if snakemake.config["h2boiler_retrofit"]:
        df = n.links[
            n.links.carrier.str.contains("gas boiler") & (n.links.build_year >= years[0])
        ].copy()
        df["bus0"] = df.index.str[:5] + " H2"
        df["bus2"] = df["bus3"]
        df["efficiency2"] = df["efficiency3"]
        df.index = df.index.str.replace("gas boiler", "H2 boiler")
        df["carrier"] = df.carrier.str.replace("gas boiler", "H2 boiler")
        import_components_from_dataframe(n, df, "Link")

    # add retrofit OCGT
    if snakemake.config["OCGT_retrofit"]:
        logger.info("add OCGTs which can be retrofitted to run with H2")
        df = n.links[
            (n.links.carrier =="OCGT") & (n.links.build_year >= years[0])
        ].copy()
        df["bus0"] = df.index.str[:5] + " H2"
        df["bus2"] = df["bus3"]
        df["efficiency2"] = df["efficiency3"]
        df.index = df.index.str.replace("OCGT", "OCGT H2")
        df["carrier"] = df.carrier.str.replace("OCGT", "OCGT H2")
        import_components_from_dataframe(n, df, "Link")

    # export network
    n.export_to_netcdf(snakemake.output[0])
