#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 18:23:20 2022

@author: lisa
"""
import os
os.chdir("/home/lisa/Documents/learning_curve/scripts")
from plot_summary import rename_techs
import numpy as np
import pandas as pd
from vresutils.costdata import annuity
from pypsa_learning.learning import (
    experience_curve,
    cumulative_cost_curve,
    piecewise_linear,
    get_slope,
)



# allow plotting without Xwindows
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from distutils.version import LooseVersion

pd_version = LooseVersion(pd.__version__)
agg_group_kwargs = dict(numeric_only=False) if pd_version >= "1.3" else {}

# from prepare_sector_network import co2_emissions_year

plt.style.use("seaborn")
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# pypsa names to technology data
pypsa_to_database = {
    "H2 electrolysis": "electrolysis",
    "H2 Electrolysis": "electrolysis",
    "H2 fuel cell": "fuel cell",
    "H2 Fuel Cell": "fuel cell",
    "battery charger": "battery inverter",
    "battery discharger": "battery inverter",
    "H2": "hydrogen storage underground",
    "battery": "battery storage",
    "offwind-ac": "offwind",
    "offwind-dc": "offwind",
    "DAC": "direct air capture",
    "battery charger": "battery inverter",
    "solar": "solar-utility",
}

path = "/home/lisa/Documents/learning_curve/results"
color_dict = {"1p5": "green",
              "1p7": "cornflowerblue",
              "2p0": "indianred",
              "base": "slategrey",
              "seqcost": "goldenrod",}
line_style = {
              "1p5": "--",
              "1p7": "-.",
              "2p0": ":"}
budgets = ["1p5", "1p7", "2p0"]
scenario = "learnH2xElectrolysisp0-learnsolarp0-learnonwindp0-learnoffwindp0"
filters = ["seqcost", "endogen", "base"]
#%%

def prepare_costs(cost_file, discount_rate, lifetime):
    """
    prepare cost data
    """
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

    costs["annuity"] = [
        (annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100.0)
        for i, v in costs.iterrows()
    ]
    costs["fixed"] = [
        (annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100.0)
        * v["investment"]
        for i, v in costs.iterrows()
    ]
    return costs


def prepare_costs_all_years(years):
    """
    prepares cost data for multiple years
    """
    all_costs = {}

    for year in years:
        all_costs[year] = prepare_costs(
            snakemake.input.costs + "/costs_{}.csv".format(year),
            snakemake.config["costs"]["discountrate"],
            snakemake.config["costs"]["lifetime"],
        )
    return all_costs

def historical_emissions(cts):
    """
    read historical emissions to add them to the carbon budget plot
    """
    # https://www.eea.europa.eu/data-and-maps/data/national-emissions-reported-to-the-unfccc-and-to-the-eu-greenhouse-gas-monitoring-mechanism-16
    # downloaded 201228 (modified by EEA last on 201221)

    df = pd.read_csv(snakemake.input.eea, encoding="latin-1")
    df.loc[df["Year"] == "1985-1987", "Year"] = 1986
    df["Year"] = df["Year"].astype(int)
    df = df.set_index(
        ["Year", "Sector_name", "Country_code", "Pollutant_name"]
    ).sort_index()

    e = pd.Series()
    e["electricity"] = "1.A.1.a - Public Electricity and Heat Production"
    e["residential non-elec"] = "1.A.4.b - Residential"
    e["services non-elec"] = "1.A.4.a - Commercial/Institutional"
    e["rail non-elec"] = "1.A.3.c - Railways"
    e["road non-elec"] = "1.A.3.b - Road Transportation"
    e["domestic navigation"] = "1.A.3.d - Domestic Navigation"
    e["international navigation"] = "1.D.1.b - International Navigation"
    e["domestic aviation"] = "1.A.3.a - Domestic Aviation"
    e["international aviation"] = "1.D.1.a - International Aviation"
    e["total energy"] = "1 - Energy"
    e["industrial processes"] = "2 - Industrial Processes and Product Use"
    e["agriculture"] = "3 - Agriculture"
    e["LULUCF"] = "4 - Land Use, Land-Use Change and Forestry"
    e["waste management"] = "5 - Waste management"
    e["other"] = "6 - Other Sector"
    e["indirect"] = "ind_CO2 - Indirect CO2"
    e["total wL"] = "Total (with LULUCF)"
    e["total woL"] = "Total (without LULUCF)"

    pol = ["CO2"]  # ["All greenhouse gases - (CO2 equivalent)"]
    if "GB" in cts:
        cts.remove("GB")
        cts.append("UK")

    year = np.arange(1990, 2020).tolist()

    idx = pd.IndexSlice
    co2_totals = (
        df.loc[idx[year, e.values, cts, pol], "emissions"]
        .unstack("Year")
        .rename(index=pd.Series(e.index, e.values))
    )

    co2_totals = (1 / 1e6) * co2_totals.groupby(level=0, axis=0).sum()  # Gton CO2

    co2_totals.loc["industrial non-elec"] = (
        co2_totals.loc["total energy"]
        - co2_totals.loc[
            [
                "electricity",
                "services non-elec",
                "residential non-elec",
                "road non-elec",
                "rail non-elec",
                "domestic aviation",
                "international aviation",
                "domestic navigation",
                "international navigation",
            ]
        ].sum()
    )

    emissions = co2_totals.loc["electricity"]

    opts = snakemake.config["scenario"]["sector_opts"]

    # if "T" in opts:
    emissions += co2_totals.loc[[i + " non-elec" for i in ["rail", "road"]]].sum()
    # if "H" in opts:
    emissions += co2_totals.loc[
        [i + " non-elec" for i in ["residential", "services"]]
    ].sum()
    # if "I" in opts:
    emissions += co2_totals.loc[
        [
            "industrial non-elec",
            "industrial processes",
            "domestic aviation",
            "international aviation",
            "domestic navigation",
            "international navigation",
        ]
    ].sum()
    return emissions


def plot_carbon_budget_distribution():
    """
    Plot historical carbon emissions in the EU and decarbonization path
    """
    cts = pd.Index(
        snakemake.config["countries"]
    )  # pd.read_csv(snakemake.input.countries, index_col=1)
    # cts = countries.index.dropna().str[:2].unique()
    co2_all = {}
    for budget in budgets:
        co2_emissions = pd.read_csv(
            path + "/newrates_73sn_{}/csvs/co2_emissions.csv".format(budget),
            index_col=0, header=list(range(3))
        )

        co2_emissions = co2_emissions.diff().fillna(co2_emissions.iloc[0, :])
        # convert tCO2 to Gt CO2 per year -> TODO annual emissions
        co2_emissions *= 1e-9
        # drop unnessary level
        co2_emissions_grouped = co2_emissions.droplevel(level=[0, 1], axis=1)

        co2_emissions_grouped =  co2_emissions_grouped.loc[:,(co2_emissions_grouped.columns.str.contains("73sn")
                                                              &co2_emissions_grouped.columns.str.contains("notarget")
                                                              & ~ co2_emissions_grouped.columns.str.contains("seqcost")
                                                           & ~ co2_emissions_grouped.columns.str.contains("nogridcost"))]
        co2_all[budget] = co2_emissions_grouped


    # historical emissions
    emissions = historical_emissions(cts.to_list())

    import matplotlib.gridspec as gridspec
    import seaborn as sns

    sns.set()
    sns.set_style("ticks")
    plt.style.use("seaborn-ticks")
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    scenarios = len(co2_emissions_grouped.columns)
    ls = (5*["-", "--", "-.", ":", "-", "--", "-.", ":"])[:scenarios]

    plt.figure(figsize=(10, 7))

    gs1 = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs1[0, 0])
    ax1.set_ylabel("CO$_2$ emissions \n [Gt per year]", fontsize=22)
    # ax1.set_ylim([0,5])
    ax1.set_xlim([1990, 2050 + 1])

    for budget in budgets:
        co2 = co2_all[budget]
        scen = co2.loc[:, co2.columns.str.contains(scenario)]

        ax1.fill_between(co2.index, co2.min(axis=1), co2.max(axis=1),
                        alpha=0.2, facecolor=color_dict[budget])
        name = 'Co2L-73sn-notarget-2p0-learnH2xElectrolysisp0-learnsolarp0-learnonwindp0-learnoffwindp0-MIP1'
        if name in scen.columns:
            scen = scen[[name]]
        if len(scen.columns)==1:
            scen.columns = [budget]



            scen.plot(ax=ax1, color=color_dict[budget], lw=2,
                      ls=line_style[budget])

    ax1.plot(emissions, color="black", linewidth=3, label=None)



    ax1.plot(
        [2030],
        [0.45 * emissions[1990]],
        marker="*",
        markersize=12,
        markerfacecolor="black",
        markeredgecolor="black",
    )



    ax1.plot(
        [2050],
        [0.0 * emissions[1990]],
        "ro",
        marker="*",
        markersize=12,
        markerfacecolor="black",
        markeredgecolor="black",
        label="EU commited target",
    )

    lg = ax1.legend(title="CO$_2$ Budgets",
        fancybox=True, fontsize=18, loc="lower left", facecolor="white", frameon=True
    )
    lg.get_title().set_fontsize(22)
    ax1.set_xlabel("year", fontsize=22)

    plt.savefig("/home/lisa/Documents/learning_curve/graphs_iew/carbon_budget_plot.pdf",
                bbox_inches="tight")


def plot_capacities():
    capacities_all = {}
    for budget in budgets:
        capacities = pd.read_csv(path + "/newrates_73sn_{}/csvs/capacities.csv".format(budget),
                                 index_col=[0, 1], header=list(range(n_header))
        )

        capacities = round(capacities.droplevel(level=[0, 1], axis=1)) / 1e3

        capacities = (
            capacities.rename(index=lambda x: rename_techs(x), level=1)
            .groupby(level=1)
            .sum()
        )

        capacities =  capacities.loc[:,(capacities.columns.get_level_values(0).str.contains("73sn-notarget")
                                        & ~ capacities.columns.get_level_values(0).str.contains("seqcost")
                                        & ~ capacities.columns.get_level_values(0).str.contains("nogridcost"))]
        capacities = capacities.loc[:,capacities.sum()!=0]

        capacities_all[budget] = capacities

    # plot all capacities
    carriers = ["solar PV", "onshore wind", "offshore wind", "H2 Electrolysis",
                "battery storage", "DAC"]

    for carrier in carriers:
        fig, ax = plt.subplots()

        for budget in budgets:
            caps = capacities_all[budget].loc[carrier].unstack().T
            scen = caps.loc[:, caps.columns.str.contains(scenario)]

            name = 'Co2L-73sn-notarget-2p0-learnH2xElectrolysisp0-learnsolarp0-learnonwindp0-learnoffwindp0-MIP1'
            if name in scen.columns:
                scen = scen[[name]]
            if len(scen.columns)==1:
                scen.columns = [budget]
                scen.plot(ax=ax, color=color_dict[budget], lw=2,
                          ls=line_style[budget])


            ax.fill_between(caps.index, caps.min(axis=1), caps.max(axis=1),
                            alpha=0.2, facecolor=color_dict[budget])



        fig.suptitle(carrier, fontsize=16)
        plt.ylabel("capacity \n [GW]")
        plt.xlabel("year")
        ax.set_xlim(["2020", "2050"])
        ax.set_ylim(bottom=0)
        plt.legend(loc="upper left", title="CO$_2$ Budgets", fancybox=True,
                   facecolor="white", frameon=True)


        plt.savefig("/home/lisa/Documents/learning_curve/graphs_iew/" + "_together_{}.pdf".format(carrier),
                    bbox_inches="tight")


def plot_capital_costs_learning():
    cost_learning_all = {}
    for budget in budgets:

        cost_learning = pd.read_csv(path + "/newrates_73sn_{}/csvs/capital_costs_learning.csv".format(budget),
                                 index_col=0, header=list(range(n_header))
        )


    # non learning assets
        capital_cost = pd.read_csv(path + "/newrates_73sn_{}/csvs/capital_cost.csv".format(budget),
                                   index_col=0,
                                   header=list(range(n_header)))
        offwind = capital_cost.reindex(["offwind-ac", "offwind-dc"]).mean()
        capital_cost.loc["offwind"].fillna(offwind, inplace=True)



        capital_cost = capital_cost.droplevel(level=[0,1], axis=1) / 1e3
        cost_learning_all[budget] = capital_cost.loc[:,(capital_cost.columns.get_level_values(0).str.contains("73sn-notarget")
                                        & ~ capital_cost.columns.get_level_values(0).str.contains("seqcost")
                                        & ~ capital_cost.columns.get_level_values(0).str.contains("nogridcost"))]

    learn_i = cost_learning.index
    years = capital_cost.columns.levels[1]
    costs = prepare_costs_all_years(years)
    # convert EUR/MW in EUR/kW
    costs_dea = (
        pd.concat(
            [
                costs[year]
                .loc[cost_learning.rename(index=pypsa_to_database).index, "fixed"]
                .rename(year)
                for year in years
            ],
            axis=1,
        )
        / 1e3
    )
    costs_dea.rename(index=lambda x: "DEA " + x, inplace=True)

    annuity_df =  (
        pd.concat(
            [
                costs[year]
                .loc[cost_learning.rename(index=pypsa_to_database).index, "annuity"]
                .rename(year)
                for year in years
            ],
            axis=1,
        ))
    annuity_df.rename(index=lambda x: "DEA " + x, inplace=True)

    investment = (
        pd.concat(
            [
                costs[year]
                .loc[cost_learning.rename(index=pypsa_to_database).index, "investment"]
                .rename(year)
                for year in years
            ],
            axis=1,
        ) /1e3)
    investment.rename(index=lambda x: "DEA " + x, inplace=True)

    for budget in budgets:

        for tech in learn_i:
            fig, ax = plt.subplots()
            dea_name = (
                 "DEA " + cost_learning.loc[[tech]].rename(pypsa_to_database).index
             )
            c = costs_dea.loc[dea_name]
            c.index = ["DEA"]
            c.T.plot(
                 ax=ax,
                 #legend=,
                 ls=(0, (3, 1, 1, 1, 1, 1)),
                 color="black", # [snakemake.config["plotting"]["tech_colors"][tech]],
            )
            # for budget in budgets:
            caps = cost_learning_all[budget].loc[tech].unstack().T
            scen = caps.loc[:, caps.columns.str.contains(scenario)]

            ax.fill_between(caps.index, caps.min(axis=1), caps.max(axis=1),
                            alpha=0.2, facecolor=color_dict[budget])
            name = 'Co2L-73sn-notarget-2p0-learnH2xElectrolysisp0-learnsolarp0-learnonwindp0-learnoffwindp0-MIP1'
            if name in scen.columns:
                scen = scen[[name]]

            if len(scen.columns)==1:
                scen.columns = [budget]

                scen.plot(ax=ax, color=color_dict[budget], lw=2,
                          ls=line_style[budget])

            fig.suptitle(tech, fontsize=16)
            plt.ylabel("annualised investment costs \n [EUR/kW per a]")
            plt.xlabel("year")
            ax.set_xlim(["2020", "2050"])
            ax.set_ylim(bottom=0)
            # plt.legend(loc="upper left", title="CO2 Budget")

            plt.savefig("/home/lisa/Documents/learning_curve/graphs_iew/" + "annualised_investment_cost_{}_{}.pdf".format(budget, tech),
                        bbox_inches="tight")

    for budget in budgets:

        for tech in learn_i:
            fig, ax = plt.subplots()
            dea_name = (
                 "DEA " + cost_learning.loc[[tech]].rename(pypsa_to_database).index
             )
            c = investment.loc[dea_name]
            c.index = ["DEA"]
            c.T.plot(
                 ax=ax,
                 #legend=,
                 ls=(0, (3, 1, 1, 1, 1, 1)),
                 color="black", # [snakemake.config["plotting"]["tech_colors"][tech]],
            )
            a = annuity_df.loc[dea_name]
            # for budget in budgets:
            caps = cost_learning_all[budget].loc[tech].unstack().T
            caps = caps.div(a.T.values)
            scen = caps.loc[:, caps.columns.str.contains(scenario)]

            ax.fill_between(caps.index, caps.min(axis=1), caps.max(axis=1),
                            alpha=0.2, facecolor=color_dict[budget])
            name = 'Co2L-73sn-notarget-2p0-learnH2xElectrolysisp0-learnsolarp0-learnonwindp0-learnoffwindp0-MIP1'
            if name in scen.columns:
                scen = scen[[name]]
            if len(scen.columns)==1:
                scen.columns = [budget]

                scen.plot(ax=ax, color=color_dict[budget], lw=2,
                          ls=line_style[budget])

            fig.suptitle(tech, fontsize=16)
            plt.ylabel("investment costs \n [EUR/kW]")
            plt.xlabel("year")
            ax.set_xlim(["2020", "2050"])
            ax.set_ylim(bottom=0)
            # plt.legend(loc="upper left", title="CO2 Budget")

            plt.savefig("/home/lisa/Documents/learning_curve/graphs_iew/" + "investment_cost_{}_{}.pdf".format(budget, tech),
                        bbox_inches="tight")

    for tech in learn_i:
        fig, ax = plt.subplots()
        dea_name = (
             "DEA " + cost_learning.loc[[tech]].rename(pypsa_to_database).index
         )
        c = investment.loc[dea_name]
        c.index = ["DEA"]
        c.T.plot(
             ax=ax,
             #legend=,
             ls=(0, (3, 1, 1, 1, 1, 1)),
             color="black", # [snakemake.config["plotting"]["tech_colors"][tech]],
        )
        a = annuity_df.loc[dea_name]
        for budget in budgets:
            caps = cost_learning_all[budget].loc[tech].unstack().T
            caps = caps.div(a.T.values)
            scen = caps.loc[:, caps.columns.str.contains(scenario)]
            ax.fill_between(caps.index, caps.min(axis=1), caps.max(axis=1),
                            alpha=0.2, facecolor=color_dict[budget])
            name = 'Co2L-73sn-notarget-2p0-learnH2xElectrolysisp0-learnsolarp0-learnonwindp0-learnoffwindp0-MIP1'
            if name in scen.columns:
                scen = scen[[name]]
            if len(scen.columns)==1:
                scen.columns = [budget]

                scen.plot(ax=ax, color=color_dict[budget], lw=2,
                          ls=line_style[budget])
        fig.suptitle(tech, fontsize=16)
        plt.ylabel("investment costs \n [EUR/kW]")
        plt.xlabel("year")
        ax.set_xlim(["2020", "2050"])
        ax.set_ylim(bottom=0)
        plt.legend(title="CO$_2$ Budgets", fancybox=True,
                   facecolor="white", frameon=True)

        plt.savefig("/home/lisa/Documents/learning_curve/graphs_iew/" + "investment_cost_together_{}.pdf".format(tech),
                    bbox_inches="tight")



def plot_costs():

    cost_df_all = {}
    for budget in budgets:

        cost_learning = pd.read_csv(path + "/newrates_73sn_{}/csvs/capital_costs_learning.csv".format(budget),
                                 index_col=0, header=list(range(n_header))
        )

        cost_df = pd.read_csv(
            path + "/newrates_73sn_{}/csvs/costs.csv".format(budget),
            index_col=list(range(3)), header=list(range(4))
        )

        df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

        df = df.loc[:,(df.columns.get_level_values(2).str.contains("73sn-notarget")
                       & ~ df.columns.get_level_values(2).str.contains("seqcost")
                       & ~ df.columns.get_level_values(2).str.contains("nogridcost"))]

        # df.rename(columns={"37": "DE"}, inplace=True)

        # convert to billions
        df = df / 1e9

        df = df.groupby(df.index.map(rename_techs)).sum()

        to_drop = df.index[df.max(axis=1) < snakemake.config["plotting"]["costs_threshold"]]

        print("dropping")

        print(df.loc[to_drop])

        df = df.drop(to_drop)

        print(df.sum())

        cost_df_all[budget] = df


    fig, ax = plt.subplots()
    for budget in budgets:
        caps = cost_df_all[budget].sum().unstack().T
        caps = caps.droplevel([0,1], axis=1)
        caps = caps.loc[:,caps.sum()!=0]
        scen = caps.loc[:, caps.columns.str.contains(scenario)]
        ax.fill_between(caps.index, caps.min(axis=1), caps.max(axis=1),
                        alpha=0.2, facecolor=color_dict[budget])
        name = 'Co2L-73sn-notarget-2p0-learnH2xElectrolysisp0-learnsolarp0-learnonwindp0-learnoffwindp0-MIP1'
        if name in scen.columns:
            scen = scen[[name]]
        if len(scen.columns)==1:
            scen.columns = [budget]

            scen.plot(ax=ax, color=color_dict[budget], lw=2,
                      ls=line_style[budget])


    plt.legend(title="CO$_2$ Budgets", fancybox=True,
               facecolor="white", frameon=True, loc="lower left")
    plt.ylabel("System Cost \n [EUR billion per year]")
    ax.set_xlim(["2020", "2050"])

    plt.savefig("/home/lisa/Documents/learning_curve/graphs_iew/total_costs_per_year.pdf",
                bbox_inches="tight")

# ############################################################################
# METHOD PART ###############################################################

def plot_costs_methods():

    cost_df_all = {}
    for budget in budgets:

        cost_learning = pd.read_csv(path + "/newrates_73sn_{}/csvs/capital_costs_learning.csv".format(budget),
                                 index_col=0, header=list(range(n_header))
        )

        cost_df = pd.read_csv(
            path + "/newrates_73sn_{}/csvs/costs.csv".format(budget),
            index_col=list(range(3)), header=list(range(4))
        )

        df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

        df = df.loc[:,(df.columns.get_level_values(2).str.contains("73sn")
                       & ~df.columns.get_level_values(2).str.contains("nogridcost"))]


        # convert to billions
        df = df / 1e9

        df = df.groupby(df.index.map(rename_techs)).sum()

        to_drop = df.index[df.max(axis=1) < snakemake.config["plotting"]["costs_threshold"]]

        print("dropping")

        print(df.loc[to_drop])

        df = df.drop(to_drop)

        print(df.sum())

        cost_df_all[budget] = df



    for budget in budgets:
        fig, ax = plt.subplots()
        caps = cost_df_all[budget].sum().unstack().T
        caps = caps.droplevel([0,1], axis=1)
        caps = caps.loc[:,caps.sum()!=0]
        caps.rename(columns = lambda x: x + "-endogen" if ("learn" in x and not "seqcost" in x) else x, inplace=True)
        caps.rename(columns = lambda x: x + "-base" if not "learn" in x else x, inplace=True)


        for sc in filters:
            df = caps.loc[:,caps.columns.str.contains(sc)]
            # df = df.loc[:, df.columns.str.contains("notarget")]
            if df.empty: continue
            if not sc=="base":
                df = df.loc[:, df.columns.str.contains("notarget")]
                scen =  df.loc[:, (df.columns.str.contains(scenario) & ~df.columns.str.contains("nogridcost"))]
                name = 'Co2L-73sn-notarget-2p0-learnH2xElectrolysisp0-learnsolarp0-learnonwindp0-learnoffwindp0-MIP1'
                if name in scen.columns:
                    scen = scen[[name]]
                if len(scen.columns)!=1:
                    print("warning {}".format(scen.columns))
                    scen = scen.iloc[:,:1]
                scen.columns = [sc]
                if sc=="endogen": sc=budget
                ax.fill_between(df.index, df.min(axis=1), df.max(axis=1),
                                alpha=0.2, facecolor=color_dict[sc])
                scen.plot(ax=ax, color=color_dict[sc], lw=2,
                          ls=line_style[budget])
            else:
                df.columns=["exogen"]
                df.plot(ax=ax, color=color_dict[sc], lw=2,
                          ls=(0, (3, 1, 1, 1, 1, 1)))


        plt.legend(title="CO$_2$ Budget {}".format(budget), fancybox=True,
                   facecolor="white", frameon=True, loc="lower left")
        plt.ylabel("System Cost \n [EUR billion per year]")
        ax.set_xlim(["2020", "2050"])

        plt.savefig("/home/lisa/Documents/learning_curve/graphs_iew/total_costs_per_year_method_{}.pdf".format(budget),
                    bbox_inches="tight")


def plot_capacities_methods():
    capacities_all = {}
    for budget in budgets:
        capacities = pd.read_csv(path + "/newrates_73sn_{}/csvs/capacities.csv".format(budget),
                                 index_col=[0, 1], header=list(range(n_header))
        )

        capacities = round(capacities.droplevel(level=[0, 1], axis=1)) / 1e3

        capacities = (
            capacities.rename(index=lambda x: rename_techs(x), level=1)
            .groupby(level=1)
            .sum()
        )

        capacities = capacities.loc[:,capacities.sum()!=0]
        capacities = capacities.loc[:, (capacities.columns.get_level_values(0).str.contains("73sn"))]
                                        #&capacities.columns.get_level_values(0).str.contains("notarget"))]

        capacities_all[budget] = capacities

    # plot all capacities
    carriers = ["solar PV", "onshore wind", "offshore wind", "H2 Electrolysis",
                "battery storage", "DAC"]

    for carrier in carriers:
        for budget in budgets:
            fig, ax = plt.subplots()
            caps = capacities_all[budget].loc[carrier].unstack().T
            caps.rename(columns = lambda x: x + "-endogen" if ("learn" in x and not "seqcost" in x) else x, inplace=True)
            caps.rename(columns = lambda x: x + "-base" if not "learn" in x else x, inplace=True)
            for sc in filters:
                df = caps.loc[:,caps.columns.str.contains(sc)]
                if df.empty: continue
                if not sc=="base":
                    scen =  df.loc[:, (df.columns.str.contains(scenario) & ~df.columns.str.contains("nogridcost"))]
                    name = 'Co2L-73sn-notarget-2p0-learnH2xElectrolysisp0-learnsolarp0-learnonwindp0-learnoffwindp0-MIP1'
                    if name in scen.columns:
                        scen = scen[[name]]
                    if len(scen.columns)==1:
                        scen.columns = [sc]
                        if sc=="endogen": sc=budget
                        scen.plot(ax=ax, color=color_dict[sc], lw=2,
                                  ls=line_style[budget])
                    if sc=="endogen": sc=budget
                    ax.fill_between(df.index, df.min(axis=1), df.max(axis=1),
                                    alpha=0.2, facecolor=color_dict[sc])
                else:
                    df.columns=["exogen"]
                    df.plot(ax=ax, color=color_dict[sc], lw=2,
                              ls=(0, (3, 1, 1, 1, 1, 1)))


            fig.suptitle(carrier, fontsize=16)
            plt.ylabel("capacity \n [GW]")
            plt.xlabel("year")
            ax.set_xlim(["2020", "2050"])
            ax.set_ylim(bottom=0)
            plt.legend(loc="upper left", title="CO$_2$ Budget {}".format(budget), fancybox=True,
                       facecolor="white", frameon=True)


            plt.savefig("/home/lisa/Documents/learning_curve/graphs_iew/" + "capacity_method_{}_{}.pdf".format(budget, carrier),
                        bbox_inches="tight")

def plot_carbon_budget_distribution_methods():
    """
    Plot historical carbon emissions in the EU and decarbonization path
    """
    cts = pd.Index(
        snakemake.config["countries"]
    )  # pd.read_csv(snakemake.input.countries, index_col=1)
    # cts = countries.index.dropna().str[:2].unique()
    co2_all = {}
    for budget in budgets:
        co2_emissions = pd.read_csv(
            path + "/newrates_73sn_{}/csvs/co2_emissions.csv".format(budget),
            index_col=0, header=list(range(3))
        )

        co2_emissions = co2_emissions.diff().fillna(co2_emissions.iloc[0, :])
        # convert tCO2 to Gt CO2 per year -> TODO annual emissions
        co2_emissions *= 1e-9
        # drop unnessary level
        co2_emissions_grouped = co2_emissions.droplevel(level=[0, 1], axis=1)

        co2_emissions_grouped =  co2_emissions_grouped.loc[:,(co2_emissions_grouped.columns.str.contains("73sn")
                                                              &co2_emissions_grouped.columns.str.contains("notarget")
                                                           & ~ co2_emissions_grouped.columns.str.contains("nogridcost"))]
        co2_all[budget] = co2_emissions_grouped


    # historical emissions
    emissions = historical_emissions(cts.to_list())

    import matplotlib.gridspec as gridspec
    import seaborn as sns

    sns.set()
    sns.set_style("ticks")
    plt.style.use("seaborn-ticks")
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20


    for budget in budgets:

        co2 = co2_all[budget]
        co2.rename(columns = lambda x: x + "-endogen" if ("learn" in x and not "seqcost" in x) else x, inplace=True)
        co2.rename(columns = lambda x: x + "-base" if not "learn" in x else x, inplace=True)

        plt.figure(figsize=(10, 7))

        gs1 = gridspec.GridSpec(1, 1)
        ax1 = plt.subplot(gs1[0, 0])
        ax1.set_ylabel("CO$_2$ emissions \n [Gt per year]", fontsize=22)
        # ax1.set_ylim([0,5])
        ax1.set_xlim([1990, 2050 + 1])


        for sc in filters:
            df = co2.loc[:,co2.columns.str.contains(sc)]
            if df.empty: continue
            if not sc=="base":
                scen =  df.loc[:, (df.columns.str.contains(scenario) & ~df.columns.str.contains("nogridcost"))]
                name = 'Co2L-73sn-notarget-2p0-learnH2xElectrolysisp0-learnsolarp0-learnonwindp0-learnoffwindp0-MIP1'
                if name in scen.columns:
                    scen = scen[[name]]

                if len(scen.columns)==1:
                    scen.columns = [sc]

                if sc=="endogen": sc=budget
                ax1.fill_between(df.index, df.min(axis=1), df.max(axis=1),
                                alpha=0.2, facecolor=color_dict[sc])
                name = 'Co2L-73sn-notarget-2p0-learnH2xElectrolysisp0-learnsolarp0-learnonwindp0-learnoffwindp0-MIP1'
                if name in scen.columns:
                    scen = scen[[name]]
                if len(scen.columns)==1:
                    scen.plot(ax=ax1, color=color_dict[sc], lw=2,
                              ls="--",
                              )

            else:
                df.columns=["exogen"]
                df.plot(ax=ax1, color=color_dict[sc], lw=2,
                          ls=(0, (3, 1, 1, 1, 1, 1))
                          )




        ax1.plot(emissions, color="black", linewidth=3, label=None)



        ax1.plot(
            [2030],
            [0.45 * emissions[1990]],
            marker="*",
            markersize=12,
            markerfacecolor="black",
            markeredgecolor="black",
        )



        ax1.plot(
            [2050],
            [0.0 * emissions[1990]],
            "ro",
            marker="*",
            markersize=12,
            markerfacecolor="black",
            markeredgecolor="black",
            label="EU commited target",
        )

        lg = ax1.legend(title="CO$_2$ budget {}".format(budget),
            fancybox=True, fontsize=14, loc="lower left", facecolor="white", frameon=True
        )
        lg.get_title().set_fontsize(16)
        ax1.set_xlabel("year", fontsize=22)

        plt.savefig("/home/lisa/Documents/learning_curve/graphs_iew/carbon_budget_plot_methods_{}.pdf".format(budget),
                    bbox_inches="tight")


def plot_capital_costs_learning_methods():
    cost_learning_all = {}
    for budget in budgets:

        cost_learning = pd.read_csv(path + "/newrates_73sn_{}/csvs/capital_costs_learning.csv".format(budget),
                                 index_col=0, header=list(range(n_header))
        )


        # non learning assets
        capital_cost = pd.read_csv(path + "/newrates_73sn_{}/csvs/capital_cost.csv".format(budget),
                                   index_col=0,
                                   header=list(range(n_header)))
        offwind = capital_cost.reindex(["offwind-ac", "offwind-dc"]).mean()
        capital_cost.loc["offwind"].fillna(offwind, inplace=True)



        capital_cost = capital_cost.droplevel(level=[0,1], axis=1) / 1e3
        cost_learning_all[budget] = capital_cost.loc[:,(capital_cost.columns.get_level_values(0).str.contains("73sn")
                                                        # & capital_cost.columns.get_level_values(0).str.contains("notarget")
                                        & ~ capital_cost.columns.get_level_values(0).str.contains("nogridcost"))]

    learn_i = cost_learning.index
    years = capital_cost.columns.levels[1]
    costs = prepare_costs_all_years(years)
    # convert EUR/MW in EUR/kW
    costs_dea = (
        pd.concat(
            [
                costs[year]
                .loc[cost_learning.rename(index=pypsa_to_database).index, "fixed"]
                .rename(year)
                for year in years
            ],
            axis=1,
        )
        / 1e3
    )
    costs_dea.rename(index=lambda x: "DEA " + x, inplace=True)

    annuity_df =  (
        pd.concat(
            [
                costs[year]
                .loc[cost_learning.rename(index=pypsa_to_database).index, "annuity"]
                .rename(year)
                for year in years
            ],
            axis=1,
        ))
    annuity_df.rename(index=lambda x: "DEA " + x, inplace=True)

    investment = (
        pd.concat(
            [
                costs[year]
                .loc[cost_learning.rename(index=pypsa_to_database).index, "investment"]
                .rename(year)
                for year in years
            ],
            axis=1,
        ) /1e3)
    investment.rename(index=lambda x: "DEA " + x, inplace=True)

    for budget in budgets:

        for tech in learn_i:

            dea_name = (
                 "DEA " + cost_learning.loc[[tech]].rename(pypsa_to_database).index
             )
            c = investment.loc[dea_name]
            c.index = ["exogen (DEA)"]
            a = annuity_df.loc[dea_name]
            # for budget in budgets:
            caps = cost_learning_all[budget].loc[tech].unstack().T
            caps = caps.div(a.T.values)

            caps.rename(columns = lambda x: x + "-endogen" if ("learn" in x and not "seqcost" in x) else x, inplace=True)
            caps.rename(columns = lambda x: x + "-base" if not "learn" in x else x, inplace=True)

            fig, ax = plt.subplots()
            c.T.plot(
                  ax=ax,
                  #legend=,
                  ls=(0, (3, 1, 1, 1, 1, 1)),
                  color="black", # [snakemake.config["plotting"]["tech_colors"][tech]],
            )

            for sc in filters:
                df = caps.loc[:,caps.columns.str.contains(sc)]
                if df.empty: continue
                if not sc=="base":
                    scen =  df.loc[:, (df.columns.str.contains(scenario) & ~df.columns.str.contains("nogridcost"))]
                    name = 'Co2L-73sn-notarget-2p0-learnH2xElectrolysisp0-learnsolarp0-learnonwindp0-learnoffwindp0-MIP1'
                    if name in scen.columns:
                        scen = scen[[name]]
                    if len(scen.columns)==1:
                        scen.columns = [sc]
                    if sc=="endogen": sc=budget
                    # ax.fill_between(df.index, df.min(axis=1), df.max(axis=1),
                    #                 alpha=0.2, facecolor=color_dict[sc])
                    if len(scen.columns)==1:
                        scen.plot(ax=ax, color=color_dict[sc], lw=2,
                                  ls=line_style[budget])



            fig.suptitle(tech, fontsize=16)
            plt.ylabel("investment costs \n [EUR/kW]")
            plt.xlabel("year")
            # ax.set_xlim(["2020", "2050"])
            # ax.set_ylim(bottom=0)
            lg = ax.legend(title="CO$_2$ budget {}".format(budget),
                fancybox=True, fontsize=14, loc="upper right", facecolor="white", frameon=True
            )
            lg.get_title().set_fontsize(16)

            plt.savefig("/home/lisa/Documents/learning_curve/graphs_iew/" + "investment_cost_methods_{}_{}.pdf".format(budget, tech),
                        bbox_inches="tight")



#%%
if "snakemake" not in globals():
    import os
    os.chdir("/home/lisa/Documents/learning_curve/scripts")
    # os.chdir("/home/lisa/mnt/lisa/learning_curve/scripts")
    from vresutils import Dict
    import yaml
    snakemake = Dict()
    # with open('/home/lisa/mnt/lisa/learning_curve/results/seqcost_1p5/configs/config.yaml', encoding='utf8') as f:
    with open('/home/lisa/Documents/learning_curve/results/testing_co2neutral/configs/config.yaml', encoding='utf8') as f:
        snakemake.config = yaml.safe_load(f)
        config  = snakemake.config
    #overwrite some options
    snakemake.input = Dict(
    costs_csv="results"  + '/' + config['run'] + '/csvs/costs.csv',
    costs="data/costs/",
    # energy="results"  + '/' + config['run'] + '/csvs/energy.csv',
    balances="results"  + '/' + config['run'] + '/csvs/supply_energy.csv',
    eea ="data/eea/UNFCCC_v24.csv",
    countries="results"  + '/' + config['run'] + '/csvs/nodal_capacities.csv',
    co2_emissions="results"  + '/' + config['run'] + '/csvs/co2_emissions.csv',
    capital_costs_learning="results"  + '/' + config['run'] + '/csvs/capital_costs_learning.csv',
    capacities="results"  + '/' + config['run'] + '/csvs/capacities.csv',
    cumulative_capacities="results"  + '/' + config['run'] + '/csvs/cumulative_capacities.csv',
    learn_carriers="results"  + '/' + config['run'] + '/csvs/learn_carriers.csv',
    capital_cost="results"  + '/' + config['run'] + '/csvs/capital_cost.csv',)

    snakemake.output = Dict(
    costs1="results"  + '/' + config['run'] + '/graphs/costs.pdf',
    costs2="results"  + '/' + config['run'] + '/graphs/costs2.pdf',
    costs3="results"  + '/' + config['run'] + '/graphs/total_costs_per_year.pdf',
    # energy="results"  + '/' + config['run'] + '/graphs/energy.pdf',
    balances="results"  + '/' + config['run'] + '/graphs/balances-energy.pdf',
    co2_emissions="results"  + '/' + config['run'] + '/graphs/carbon_budget_plot.pdf',
    capacities="results"  + '/' + config['run'] + '/graphs/capacities.pdf',
    capital_costs_learning="results"  + '/' + config['run'] + '/graphs/capital_costs_learning.pdf',
    annual_investments="results"  + '/' + config['run'] + '/graphs/annual_investments.pdf',
    learning_cost_vs_curve="results"  + '/' + config['run'] + '/graphs/learning_cost_vs_curve/learning_cost.pdf',
    )
    os.chdir("/home/lisa/Documents/learning_curve/")

n_header=4

plot_carbon_budget_distribution()
plot_costs()
plot_capital_costs_learning()
plot_capacities()

plot_costs_methods()
plot_capacities_methods()
plot_carbon_budget_distribution_methods()
