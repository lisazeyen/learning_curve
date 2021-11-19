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
}

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


# consolidate and rename
def rename_techs(label):

    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral ",
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        # "H2" : "hydrogen storage",
        "battery": "battery storage",
        # "CC" : "CC"
    }

    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "offwind-ac": "offshore wind (AC)",
        "offwind-dc": "offshore wind (DC)",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines",
        "H2": "hydrogen storage",
    }

    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old, new in rename.items():
        if old == label:
            label = new
    return label


preferred_order = pd.Index(
    [
        "transmission lines",
        "hydroelectricity",
        "hydro reservoir",
        "run of river",
        "pumped hydro storage",
        "solid biomass",
        "biogas",
        "onshore wind",
        "offshore wind",
        "offshore wind (AC)",
        "offshore wind (DC)",
        "solar PV",
        "solar thermal",
        "solar",
        "building retrofitting",
        "ground heat pump",
        "air heat pump",
        "heat pump",
        "resistive heater",
        "power-to-heat",
        "gas-to-power/heat",
        "CHP",
        "OCGT",
        "gas boiler",
        "gas",
        "natural gas",
        "helmeth",
        "methanation",
        "hydrogen storage",
        "power-to-gas",
        "power-to-liquid",
        "battery storage",
        "hot water storage",
        "CO2 sequestration",
    ]
)


def plot_costs():

    cost_df = pd.read_csv(
        snakemake.input.costs_csv, index_col=list(range(3)), header=list(range(4))
    )

    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    # df.rename(columns={"37": "DE"}, inplace=True)

    # convert to billions
    df = df / 1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.max(axis=1) < snakemake.config["plotting"]["costs_threshold"]]

    print("dropping")

    print(df.loc[to_drop])

    df = df.drop(to_drop)

    print(df.sum())

    new_index = preferred_order.intersection(df.index).append(
        df.index.difference(preferred_order)
    )

    new_columns = df.sum().sort_values().index

    # PLOT 1 ##############################################################
    fig, ax = plt.subplots()
    fig.set_size_inches((12, 8))

    df.loc[new_index].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=[snakemake.config["plotting"]["tech_colors"][i] for i in new_index],
    )

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    # ax.set_ylim([0,snakemake.config['plotting']['costs_max']])

    ax.set_ylabel("System Cost [EUR billion per year]")

    ax.set_xlabel("")

    ax.grid(axis="y")

    ax.legend(handles, labels, ncol=4, loc="upper left", bbox_to_anchor=(0.01, 1.25))

    fig.savefig(snakemake.output.costs1, transparent=True, bbox_inches="tight")

    # PLOT 2 ##############################################################
    if len(df.stack().columns) > 1:
        fig, ax = plt.subplots(len(df.stack().columns), 1, sharex=True)
        fig.set_size_inches((10, 20))
        for i, scenario in enumerate(df.stack().columns):

            df.loc[new_index, scenario].T.plot(
                kind="bar",
                ax=ax[i],
                stacked=True,
                title=str(scenario),
                legend=False,
                color=[
                    snakemake.config["plotting"]["tech_colors"][i] for i in new_index
                ],
            )

            ax[i].set_xlabel("")

            ax[i].set_ylim([0, df.sum().max() * 1.1])

            ax[i].grid(axis="y")

        ax[0].set_ylabel("System Cost [EUR billion per year]")

        handles, labels = ax[0].get_legend_handles_labels()

        handles.reverse()
        labels.reverse()

        ax[1].legend(handles, labels, ncol=1, bbox_to_anchor=(1, 1))

    fig.savefig(snakemake.output.costs2, bbox_inches="tight")

    # PLOT 3 ##############################################################
    scenarios = len(df.stack().columns)
    ls = ["-", "--", "-.", ":", "-", "--", "-.", ":"][:scenarios]

    fig, ax = plt.subplots()
    df.sum().unstack().T.plot(
        title="total annual costs", grid=True, ax=ax, lw=4, style=ls
    )
    plt.legend(bbox_to_anchor=(1, 1))
    plt.ylabel("System Cost [EUR billion per year]")

    fig.savefig(snakemake.output.costs3, bbox_inches="tight")


def plot_energy():

    energy_df = pd.read_csv(
        snakemake.input.energy, index_col=list(range(2)), header=list(range(n_header))
    )

    df = energy_df.groupby(energy_df.index.get_level_values(1)).sum()

    # convert MWh to TWh
    df = df / 1e6

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[
        df.abs().max(axis=1) < snakemake.config["plotting"]["energy_threshold"]
    ]

    print("dropping")

    print(df.loc[to_drop])

    df = df.drop(to_drop)

    print(df.sum())

    print(df)

    new_index = preferred_order.intersection(df.index).append(
        df.index.difference(preferred_order)
    )

    new_columns = df.columns.sort_values()
    # new_columns = df.sum().sort_values().index
    fig, ax = plt.subplots()
    fig.set_size_inches((12, 8))

    print(df.loc[new_index, new_columns])

    df.loc[new_index, new_columns].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=[snakemake.config["plotting"]["tech_colors"][i] for i in new_index],
    )

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim(
        [
            snakemake.config["plotting"]["energy_min"],
            snakemake.config["plotting"]["energy_max"],
        ]
    )

    ax.set_ylabel("Energy [TWh/a]")

    ax.set_xlabel("")

    ax.grid(axis="y")

    ax.legend(handles, labels, ncol=4, loc="upper left")

    fig.tight_layout()

    fig.savefig(snakemake.output.energy, transparent=True)


def plot_balances():

    co2_carriers = ["co2", "co2 stored", "process emissions"]

    balances_df = pd.read_csv(
        snakemake.input.balances, index_col=list(range(3)), header=list(range(n_header))
    )

    balances = {i.replace(" ", "_"): [i] for i in balances_df.index.levels[0]}
    balances["energy"] = [
        i for i in balances_df.index.levels[0] if i not in co2_carriers
    ]

    for k, v in balances.items():

        df = balances_df.loc[v]
        df = df.groupby(df.index.get_level_values(2)).sum()

        # convert MWh to TWh
        df = df / 1e6

        # remove trailing link ports
        df.index = [
            i[:-1]
            if ((i not in ["co2", "H2"]) and (i[-1:] in ["0", "1", "2", "3"]))
            else i
            for i in df.index
        ]

        df = df.groupby(df.index.map(rename_techs)).sum()

        to_drop = df.index[
            df.abs().max(axis=1) < snakemake.config["plotting"]["energy_threshold"] / 10
        ]

        print("dropping")

        print(df.loc[to_drop])

        df = df.drop(to_drop)

        print(df.sum())

        if df.empty:
            continue

        df = df.groupby(df.index).sum()

        new_index = preferred_order.intersection(df.index).append(
            df.index.difference(preferred_order)
        )

        new_columns = df.columns.sort_values()

        dict_re = {
            "Co2L-73sn-co2seq1": "no_learning",
            "Co2L-73sn-learnonwindp0-learnbatteryp0-learnbatteryxchargerp0-learnH2xelectrolysisp0-learnH2xFuelxCellp0-learnDACp0-learnsolarp0-co2seq1": "global learning",
            "Co2L-73sn-learnonwindp0-learnbatteryp0-learnbatteryxchargerp0-learnH2xelectrolysisp0-learnH2xFuelxCellp0-learnDACp0-learnsolarp0-co2seq1-local": "local_learning",
            "Co2L-73sn-learnonwindp0-learnbatteryp0-learnbatteryxchargerp0-learnH2xelectrolysisp10-learnH2xFuelxCellp10-learnDACp10-learnsolarp0-co2seq1": "strong_learning",
        }
        fig, ax = plt.subplots()
        fig.set_size_inches((12, 8))

        df.loc[new_index, new_columns].T.plot(
            kind="bar",
            ax=ax,
            stacked=True,
            color=[snakemake.config["plotting"]["tech_colors"][i] for i in new_index],
        )

        ax.set_ylim([df[df < 0].sum().min() * 1.1, df[df > 0].sum().max() * 1.7])

        ax.grid(axis="y", color="lightgrey")

        handles, labels = ax.get_legend_handles_labels()

        handles.reverse()
        labels.reverse()

        if v[0] in co2_carriers:
            ax.set_ylabel("CO2 [MtCO2/a]")
        else:
            ax.set_ylabel("Energy [TWh/a]")

        ax.set_xlabel("")

        ax.legend(handles, labels, ncol=4, loc="upper left")

        fig.savefig(
            snakemake.output.balances[:-10] + k + ".pdf",
            transparent=True,
            bbox_inches="tight",
        )


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

    year = np.arange(1990, 2018).tolist()

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
    co2_emissions = pd.read_csv(
        snakemake.input.co2_emissions, index_col=0, header=list(range(3))
    )

    co2_emissions = co2_emissions.diff().fillna(co2_emissions.iloc[0, :])
    # convert tCO2 to Gt CO2 per year -> TODO annual emissions
    co2_emissions *= 1e-9
    # drop unnessary level
    co2_emissions_grouped = co2_emissions.droplevel(level=[0, 1], axis=1)

    # co2_emissions_grouped = co2_emissions.stack(0).groupby(level=1).sum().T
    # co2_emissions_grouped.index = [int(x) for x in co2_emissions_grouped.index]
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
    ls = ["-", "--", "-.", ":", "-", "--", "-.", ":"][:scenarios]

    plt.figure(figsize=(10, 7))

    gs1 = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs1[0, 0])
    ax1.set_ylabel("CO$_2$ emissions (Gt per year)", fontsize=22)
    # ax1.set_ylim([0,5])
    ax1.set_xlim([1990, 2050 + 1])

    co2_emissions_grouped.plot(ax=ax1, linewidth=3, style=ls)

    ax1.plot(emissions, color="black", linewidth=3, label=None)

    # plot commited and uder-discussion targets
    # (notice that historical emissions include all countries in the
    # network, but targets refer to EU)
    ax1.plot(
        [2020],
        [0.8 * emissions[1990]],
        marker="*",
        markersize=12,
        markerfacecolor="black",
        markeredgecolor="black",
    )

    ax1.plot(
        [2030],
        [0.45 * emissions[1990]],
        marker="*",
        markersize=12,
        markerfacecolor="white",
        markeredgecolor="black",
    )

    ax1.plot(
        [2030],
        [0.6 * emissions[1990]],
        marker="*",
        markersize=12,
        markerfacecolor="black",
        markeredgecolor="black",
    )

    ax1.plot(
        [2050, 2050],
        [x * emissions[1990] for x in [0.2, 0.05]],
        color="gray",
        linewidth=2,
        marker="_",
        alpha=0.5,
    )

    ax1.plot(
        [2050],
        [0.01 * emissions[1990]],
        marker="*",
        markersize=12,
        markerfacecolor="white",
        linewidth=0,
        markeredgecolor="black",
        label="EU under-discussion target",
        zorder=10,
        clip_on=False,
    )

    ax1.plot(
        [2050],
        [0.125 * emissions[1990]],
        "ro",
        marker="*",
        markersize=12,
        markerfacecolor="black",
        markeredgecolor="black",
        label="EU commited target",
    )

    ax1.legend(
        fancybox=True, fontsize=18, loc=(1.1, 0.5), facecolor="white", frameon=True
    )

    plt.savefig(snakemake.output.co2_emissions, bbox_inches="tight")


def plot_capital_costs_learning():
    cost_learning = pd.read_csv(
        snakemake.input.capital_costs_learning,
        index_col=0,
        header=list(range(n_header)),
    )
    years = cost_learning.columns.levels[3]
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

    cost_learning = cost_learning.droplevel(level=[0, 1], axis=1) / 1e3

    for tech in cost_learning.index:
        fig, ax = plt.subplots(len(cost_learning.stack().columns), 1, sharex=True)
        fig.set_size_inches((10, 10))
        for i, scenario in enumerate(cost_learning.stack().columns):

            cost_learning.loc[tech, scenario].T.plot(
                kind="bar",
                ax=ax[i],
                title=str(scenario),
                legend=False,
                color=[snakemake.config["plotting"]["tech_colors"][tech]],
            )

            dea_name = (
                "DEA " + cost_learning.loc[[tech]].rename(pypsa_to_database).index
            )
            costs_dea.loc[dea_name].T.plot(
                ax=ax[i],
                legend=False,
                ls="--",
                color=[snakemake.config["plotting"]["tech_colors"][tech]],
            )

            ax[i].set_xlabel("")
            y_max = max(
                cost_learning.loc[tech].max(), costs_dea.loc[dea_name].max().max()
            )
            ax[i].set_ylim([0, y_max * 1.1])

            ax[i].grid(axis="y")

        ax[1].set_ylabel("Investment costs [EUR/kW]")

        handles, labels = ax[0].get_legend_handles_labels()

        handles.reverse()
        labels.reverse()

        ax[1].legend(handles, labels, ncol=1, bbox_to_anchor=(1, 1))

        fig.savefig(
            snakemake.output.capital_costs_learning[:-4] + tech + ".pdf",
            bbox_inches="tight",
        )


def learning_cost_vs_curve():
    # capital cost for learning technologies
    cost_learning = pd.read_csv(
        snakemake.input.capital_costs_learning,
        index_col=0,
        header=list(range(n_header)),
    )
    cost_learning = cost_learning.stack().unstack(0).dropna(how="all", axis=1)
    # learning technologies
    learn_i = cost_learning.index
    # cumulative local installed capacities
    cum_cap = pd.read_csv(
        snakemake.input.cumulative_capacities, index_col=0, header=list(range(n_header))
    )
    cum_cap = cum_cap.stack().unstack(0).dropna(how="all", axis=1)
    # learning carriers
    learn_carrier = pd.read_csv(
        snakemake.input.learn_carriers, index_col=0, header=list(range(n_header))
    )

    learn_carrier = learn_carrier.stack().unstack(0).dropna(how="all", axis=1)

    for scenario in learn_carrier.columns:
        initial_capacity = learn_carrier.loc["global_capacity", scenario]
        global_factor = learn_carrier.loc["global_factor", scenario]
        global_cum = (cum_cap[scenario] / global_factor) + initial_capacity
        investment_cost = cost_learning[scenario]
        tot = pd.concat([global_cum, investment_cost], axis=1)
        tot.columns = ["cap", "cost"]
        max_capacity = learn_carrier.loc["max_capacity", scenario]
        learning_rate = learn_carrier.loc["learning_rate", scenario]
        c0 = learn_carrier.loc["initial_cost", scenario]
        # for the x-axis cumulative global capacity
        caps = pd.DataFrame(
            np.arange(
                initial_capacity,
                max_capacity,
                (max_capacity - initial_capacity) // 1000,
            ),
            columns=["cumualtive capacity"],
        )
        caps.index = caps["cumualtive capacity"]
        # cumulative investment (right y-axis)
        y_cum = caps.apply(
            lambda x: cumulative_cost_curve(x, learning_rate, c0, initial_capacity)
        )
        y_cum.columns = ["cumulative investment costs"]
        # specific investment costs [Eur/MW] (left y-axis)
        y = caps.apply(
            lambda x: experience_curve(x, learning_rate, c0, initial_capacity)
        )
        y.columns = ["learning curve"]

        # get interpolation points (number of points = line segments + 1)
        points = piecewise_linear(
            caps.iloc[:, 0],
            y_cum.iloc[:, 0],
            snakemake.config["segments"],
            scenario[-1],
        )
        # get interpolation of specific investment costs
        interpolated_costs = get_slope(points).rename(index=lambda x: x + 1)
        interpolated_costs.loc[0] = c0

        # convert Eur/MW to Eur/kW
        interpolated_costs = interpolated_costs.sort_index() * 1e-3
        tot["cost"] = tot.cost.apply(lambda x: x / 1e3)

        fig, ax = plt.subplots()
        plt.title(scenario[-1])

        plt.step(
            points.xs("x_fit", level=1, axis=1),
            interpolated_costs,
            where="pre",
            lw=2,
            ls=":",
            color="green",
        )

        # plt.vlines(points[scenario[-1], "x_fit"].values, ymin=y.min()/1e3,ymax=y.max()/1e3,
        #             linestyle="-", color="grey", alpha=0.2)

        (y / 1e3).plot(ax=ax, lw=2, legend=False, color="#1f77b4")

        tot.plot(
            kind="scatter", x="cap", y="cost", ax=ax, marker="o", s=80, color="red"
        )

        ax.set_ylabel("specific annualised investment cost \n [Eur/kW]")
        ax.set_xlabel("installed global capacity \n [MW]")

        ax2 = ax.twinx()
        (y_cum / 1e9).plot(ax=ax2, lw=2, color="#1f77b4")
        ax2.set_ylabel("total cumulative cost for new capacity \n [billion Eur]")

        for lr in points.columns.levels[0]:
            lin = (
                points[lr]
                .set_index("x_fit")
                .rename(columns={"y_fit": "piece-wise linearisation"})
            )
            (lin / 1e9).plot(
                marker="*",
                ls="--",
                ax=ax2,
                grid=True,
                lw=2,
                markersize=10,
                color="green",
            )

        # tot2.plot(kind="scatter", x= 'global_cap', y="cum_cost", ax=ax2,
        #   marker="x",  s=100, color="orange", grid=True)

        plt.legend(loc="upper center")

        fig.savefig(
            snakemake.output.learning_cost_vs_curve.replace("learning_cost.pdf", "")
            + "{}.pdf".format(scenario),
            bbox_inches="tight",
        )

        fig.savefig(snakemake.output.learning_cost_vs_curve, bbox_inches="tight")


def plot_capacities():
    capacities = pd.read_csv(
        snakemake.input.capacities, index_col=[0, 1], header=list(range(n_header))
    )

    capacities = capacities.droplevel(level=[0, 1], axis=1) / 1e3

    capacities = (
        capacities.rename(index=lambda x: rename_techs(x), level=1)
        .groupby(level=1)
        .sum()
    )

    fig, ax = plt.subplots(len(capacities.stack().columns), 1, sharex=True)
    fig.set_size_inches((10, 10))

    for i, scenario in enumerate(capacities.stack().columns):

        capacities[scenario].T.plot(
            kind="bar",
            ax=ax[i],
            title=str(scenario),
            legend=False,
            color=[
                snakemake.config["plotting"]["tech_colors"][i] for i in capacities.index
            ],
        )

        ax[i].set_xlabel("")

        ax[i].set_ylim([0, capacities.max().max() * 1.1])

        ax[i].grid(axis="y")

    ax[1].set_ylabel("Installed capacities [GW]")

    handles, labels = ax[0].get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax[1].legend(handles, labels, ncol=1, bbox_to_anchor=(1, 1))

    for i, scenario in enumerate(capacities.stack().columns):
        ax[i].grid(axis="y")

    fig.savefig(snakemake.output.capacities, bbox_inches="tight")

    # #####################
    # learning carriers

    learn_carrier = pd.read_csv(
        snakemake.input.learn_carriers, index_col=0, header=list(range(n_header))
    ).index
    for carrier in learn_carrier:

        fig, ax = plt.subplots(len(capacities.stack().columns), 1, sharex=True)
        fig.suptitle(carrier, fontsize=16)
        fig.set_size_inches((10, 10))

        for i, scenario in enumerate(capacities.stack().columns):

            capacities.reindex(index=[rename_techs(carrier)])[scenario].T.plot(
                kind="area",
                ax=ax[i],
                title=str(scenario),
                legend=False,
                grid=True,
                color=[
                    snakemake.config["plotting"]["tech_colors"][i] for i in [carrier]
                ],
            )

            ax[i].set_xlabel("")

            ax[i].set_ylim(
                [0, capacities.reindex(index=[rename_techs(carrier)]).max().max() * 1.1]
            )
            # ax[i].set_xlim(["2020", "2050"])

            ax[i].grid(axis="y")

        ax[1].set_ylabel("Installed capacities [GW]")

        handles, labels = ax[0].get_legend_handles_labels()

        handles.reverse()
        labels.reverse()

        ax[1].legend(handles, labels, ncol=1)

        for i, scenario in enumerate(capacities.stack().columns):
            ax[i].grid(axis="y")
            ax[i].autoscale(enable=True, axis="x", tight=True)
        plt.savefig(
            snakemake.output.capacities[:-4] + "-{}.pdf".format(carrier),
            bbox_inches="tight",
        )


#%%
if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        import os

        os.chdir("/home/lisa/mnt/lisa/learning_curve/scripts")
        # os.chdir("/home/lisa/Documents/learning_curve/scripts")
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_summary",
            sector_opts="Co2L-876h-learnsolarp0-learnonwindp10",
            clusters="37",
        )

    n_header = 4

    plot_costs()

    plot_carbon_budget_distribution()

    # plot_energy()

    plot_balances()

    plot_capital_costs_learning()

    learning_cost_vs_curve()

    plot_capacities()
