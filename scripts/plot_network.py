import pypsa_learning as pypsa

from pypsa_learning.descriptors import  get_active_assets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Ellipse

from make_summary2 import assign_carriers
from plot_summary import rename_techs, preferred_order

from distutils.version import LooseVersion
pd_version = LooseVersion(pd.__version__)
agg_group_kwargs = dict(numeric_only=False) if pd_version >= "1.3" else {}


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
override_component_attrs["StorageUnit"].loc["p_dispatch"] = [
    "series",
    "MW",
    0.0,
    "Storage discharging.",
    "Output",
]
override_component_attrs["StorageUnit"].loc["p_store"] = [
    "series",
    "MW",
    0.0,
    "Storage charging.",
    "Output",
]

plt.style.use('ggplot')


def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    elif tech in ["H2 Electrolysis", "methanation", "helmeth", "H2 liquefaction"]:
        return "power-to-gas"
    elif tech == "H2":
        return "H2 storage"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    elif "solar" in tech:
        return "solar"
    elif tech == "Fischer-Tropsch":
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif "CC" in tech or "sequestration" in tech:
        return "CCS"
    else:
        return tech


def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()

    def axes2pt():
        return np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[0] * (72. / fig.dpi)

    ellipses = []
    if not dont_resize_actively:
        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses:
                e.width, e.height = 2. * radius * dist
        fig.canvas.mpl_connect('resize_event', update_width_height)
        ax.callbacks.connect('xlim_changed', update_width_height)
        ax.callbacks.connect('ylim_changed', update_width_height)

    def legend_circle_handler(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
        w, h = 2. * orig_handle.get_radius() * axes2pt()
        e = Ellipse(xy=(0.5 * width - 0.5 * xdescent, 0.5 *
                        height - 0.5 * ydescent), width=w, height=w)
        ellipses.append((e, orig_handle.get_radius()))
        return e
    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}


def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0, 0), radius=(s / scale)**0.5, **kw) for s in sizes]


def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1: continue
            names = ifind.index[ifind == i]
            c.df.loc[names, 'location'] = names.str[:i]


def plot_map(network, components=["Link", "Store", "StorageUnit", "Generator"],
             bus_size_factor=1.7e10):

    n = network.copy()
    assign_location(n)
    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)
    # investment periods
    investments = n.snapshots.levels[0]

    costs = {}
    for comp in components:
        df_c = n.df(comp)
        if df_c.empty: continue
        df_c["nice_group"] = df_c.carrier.map(rename_techs_tyndp)

        attr = "e_nom_opt" if comp == "Store" else "p_nom_opt"

        active = pd.concat(
            [
                get_active_assets(n, comp, inv_p, n.snapshots).rename(inv_p)
                for inv_p in investments
            ],
            axis=1,
        ).astype(int)
        capital_cost = n.df(comp)[attr] * n.df(comp).capital_cost
        capital_cost_t = ((active.mul(capital_cost, axis=0))
                 .groupby([n.df(comp).location,
                           n.df(comp).nice_group]).sum(**agg_group_kwargs))

        capital_cost_t.drop("load", level=1, inplace=True, errors="ignore")

        costs[comp] = capital_cost_t

    costs = pd.concat(costs).groupby(level=[1,2]).sum()
    costs.drop(costs[costs.sum(axis=1)==0].index, inplace=True)

    new_columns = (preferred_order.intersection(costs.index.levels[1])
                   .append(costs.index.levels[1].difference(preferred_order)))
    costs = costs.reindex(new_columns, level=1)

    for item in new_columns:
        if item not in snakemake.config['plotting']['tech_colors']:
            print("Warning!",item,"not in config/plotting/tech_colors")

    # hack because impossible to drop buses...
    if "DE0 0" in n.buses.index:
        n.buses.loc["EU gas", ["x", "y"]] = n.buses.loc["DE0 0", ["x", "y"]]
    if "Central AC" in n.buses.index:
        n.buses.loc["EU gas", ["x", "y"]] = n.buses.loc["Central AC", ["x", "y"]]
        n.lines.bus0 = n.lines.bus0.str.replace(" AC", "")
        n.lines.bus1 = n.lines.bus1.str.replace(" AC", "")
        n.links.bus1 = n.links.bus1.str.replace(" AC", "")
        n.links.bus0 = n.links.bus0.str.replace(" AC", "")
        n.buses.rename(index=lambda x: x.replace(" AC", ""), inplace=True)

    n.links.drop(n.links.index[(n.links.carrier != "DC") & (
        n.links.carrier != "B2B")], inplace=True)

    # drop non-bus
    to_drop = costs.index.levels[0].symmetric_difference(n.buses.index)
    if len(to_drop) != 0:
        print("dropping non-buses", to_drop)
        costs.drop(to_drop, level=0, inplace=True, axis=0, errors="ignore")

    # make sure they are removed from index
    costs.index = pd.MultiIndex.from_tuples(costs.index.values)

    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.
    line_upper_threshold = 1e4
    linewidth_factor = 2e3
    ac_color = "gray"
    dc_color = "m"

    line_widths = n.lines.s_nom_opt
    link_widths = n.links.p_nom_opt
    linewidth_factor = 2e3
    line_lower_threshold = 0.
    title = "Today's transmission"

    line_widths[line_widths < line_lower_threshold] = 0.
    link_widths[link_widths < line_lower_threshold] = 0.

    line_widths[line_widths > line_upper_threshold] = line_upper_threshold
    link_widths[link_widths > line_upper_threshold] = line_upper_threshold

    for year in costs.columns:

        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        fig.set_size_inches(7, 6)
        fig.suptitle(year)

        n.plot(
            bus_sizes=costs[year] / bus_size_factor,
            bus_colors=snakemake.config['plotting']['tech_colors'],
            line_colors=ac_color,
            link_colors=dc_color,
            line_widths=line_widths / linewidth_factor,
            link_widths=link_widths / linewidth_factor,
            ax=ax,  **map_opts
        )

        handles = make_legend_circles_for(
            [10e9, 5e9],
            scale=bus_size_factor,
            facecolor="gray"
        )

        labels = ["{} billion EUR/a".format(s) for s in (10, 5)]

        l2 = ax.legend(
            handles, labels,
            loc="upper left",
            bbox_to_anchor=(0.001, 1.1),
            labelspacing=0.5,
            prop={'size': 10},
            frameon=False,
            title='System cost',
            handler_map=make_handler_map_to_scale_circles_as_in(ax)
        )

        ax.add_artist(l2)

        handles = []
        labels = []

        for s in (10, 5):
            handles.append(plt.Line2D([0], [0], color=ac_color,
                                      linewidth=s * 1e3 / linewidth_factor))
            labels.append("{} GW".format(s))

        l1_1 = ax.legend(
            handles, labels,
            loc="upper left",
            bbox_to_anchor=(0.001, 0.96),
            frameon=False,
            prop={'size': 10},
            labelspacing=0.5,
            handletextpad=1.5,
            title=title
        )

        ax.add_artist(l1_1)

        fig.savefig(
            snakemake.output.map[:-9] + "_{}.pdf".format(year),
            transparent=True,
            bbox_inches="tight"
        )


def plot_h2_map(network):

    n = network.copy()

    assign_location(n)

    bus_size_factor = 1e5
    linewidth_factor = 1e4
    # MW below which not drawn
    line_lower_threshold = 1e3

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    elec = n.links[n.links.carrier.isin(["H2 Electrolysis", "H2 Fuel Cell"])].index
    comp = "Link"
    investments = n.snapshots.levels[0]
    active = pd.concat(
        [
            get_active_assets(n, comp, inv_p, n.snapshots).rename(inv_p)
            for inv_p in investments
        ],
        axis=1,
    ).astype(int)
    cap = n.df(comp)["p_nom_opt"]
    cap_t = active.mul(cap, axis=0).loc[elec]
    bus_sizes = cap_t.groupby([n.links["bus0"], n.links.carrier]).sum() / bus_size_factor

    # make a fake MultiIndex so that area is correct for legend
    bus_sizes.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)
    bus_sizes.rename(index=lambda x: x.replace(" AC", ""), level=0, inplace=True)
    n.buses.rename(index=lambda x: x.replace(" AC", ""), level=0, inplace=True)

    n.links.drop(n.links.index[~n.links.carrier.str.contains("H2 pipeline")], inplace=True)

    h2_new = n.links.loc[n.links.carrier=="H2 pipeline", "p_nom_opt"]

    h2_retro = n.links.loc[n.links.carrier=='H2 pipeline retrofitted']

    positive_order = h2_retro.bus0 < h2_retro.bus1
    h2_retro_p = h2_retro[positive_order]
    swap_buses = {"bus0": "bus1", "bus1": "bus0"}
    h2_retro_n = h2_retro[~positive_order].rename(columns=swap_buses)
    h2_retro = pd.concat([h2_retro_p, h2_retro_n])

    h2_retro.index = h2_retro.apply(
        lambda x: f"H2 pipeline {x.bus0.replace(' H2', '')} -> {x.bus1.replace(' H2', '')}",
        axis=1
    )

    h2_retro = h2_retro["p_nom_opt"]

    link_widths_total = (h2_new + h2_retro) / linewidth_factor
    link_widths_total = link_widths_total.groupby(level=0).sum().reindex(n.links.index).fillna(0.)
    link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.

    retro = n.links.p_nom_opt.where(n.links.carrier=='H2 pipeline retrofitted', other=0.)
    link_widths_retro = retro / linewidth_factor
    link_widths_retro[n.links.p_nom_opt < line_lower_threshold] = 0.

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")

    for year in bus_sizes.columns:

        fig, ax = plt.subplots(
            figsize=(7, 6),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        fig.suptitle(year)

        n.plot(
            bus_sizes=bus_sizes[year],
            bus_colors=snakemake.config['plotting']['tech_colors'],
            link_colors='#a2f0f2',
            link_widths=link_widths_total,
            branch_components=["Link"],
            ax=ax,
            **map_opts
        )

        n.plot(
            geomap=False,
            bus_sizes=0,
            link_colors='#72d3d6',
            link_widths=link_widths_retro,
            branch_components=["Link"],
            ax=ax,
            **map_opts
        )

        handles = make_legend_circles_for(
            [50000, 10000],
            scale=bus_size_factor,
            facecolor='grey'
        )

        labels = ["{} GW".format(s) for s in (50, 10)]

        l2 = ax.legend(
            handles, labels,
            loc="upper left",
            bbox_to_anchor=(-0.03, 1.2),
            labelspacing=1.0,
            frameon=False,
            title='Electrolyzer capacity',
            handler_map=make_handler_map_to_scale_circles_as_in(ax)
        )

        ax.add_artist(l2)

        handles = []
        labels = []

        for s in (50, 10):
            handles.append(plt.Line2D([0], [0], color="grey",
                                      linewidth=s * 1e3 / linewidth_factor))
            labels.append("{} GW".format(s))

        l1_1 = ax.legend(
            handles, labels,
            loc="upper left",
            bbox_to_anchor=(-0.03, 0.97),
            frameon=False,
            labelspacing=0.8,
            handletextpad=1.5,
            title='H2 pipeline capacity'
        )

        ax.add_artist(l1_1)

        fig.savefig(
            snakemake.output.map.replace("-costs-all","-h2_network_{}".format(year)),
            bbox_inches="tight"
        )


def plot_ch4_map(network):

    n = network.copy()

    if "gas pipeline" not in n.links.carrier.unique():
        return

    assign_location(n)

    bus_size_factor = 8e7
    linewidth_factor = 1e4
    # MW below which not drawn
    line_lower_threshold = 500

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    fossil_gas_i = n.generators[n.generators.carrier=="gas"].index
    fossil_gas = n.generators_t.p.loc[:,fossil_gas_i].mul(n.snapshot_weightings.generators, axis=0).sum().groupby(n.generators.loc[fossil_gas_i,"bus"]).sum() / bus_size_factor
    fossil_gas.rename(index=lambda x: x.replace(" gas", ""), inplace=True)
    fossil_gas = fossil_gas.reindex(n.buses.index).fillna(0)
    # make a fake MultiIndex so that area is correct for legend
    fossil_gas.index = pd.MultiIndex.from_product([fossil_gas.index, ["fossil gas"]])

    methanation_i = n.links[n.links.carrier.isin(["helmeth", "Sabatier"])].index
    methanation = abs(n.links_t.p1.loc[:,methanation_i].mul(n.snapshot_weightings.generators, axis=0)).sum().groupby(n.links.loc[methanation_i,"bus1"]).sum() / bus_size_factor
    methanation = methanation.groupby(methanation.index).sum().rename(index=lambda x: x.replace(" gas", ""))
    # make a fake MultiIndex so that area is correct for legend
    methanation.index = pd.MultiIndex.from_product([methanation.index, ["methanation"]])

    biogas_i = n.stores[n.stores.carrier=="biogas"].index
    biogas = n.stores_t.p.loc[:,biogas_i].mul(n.snapshot_weightings.generators, axis=0).sum().groupby(n.stores.loc[biogas_i,"bus"]).sum() / bus_size_factor
    biogas = biogas.groupby(biogas.index).sum().rename(index=lambda x: x.replace(" biogas", ""))
    # make a fake MultiIndex so that area is correct for legend
    biogas.index = pd.MultiIndex.from_product([biogas.index, ["biogas"]])

    bus_sizes = pd.concat([fossil_gas, methanation, biogas])
    bus_sizes.sort_index(inplace=True)

    to_remove = n.links.index[~n.links.carrier.str.contains("gas pipeline")]
    n.links.drop(to_remove, inplace=True)

    link_widths_rem = n.links.p_nom_opt / linewidth_factor
    link_widths_rem[n.links.p_nom_opt < line_lower_threshold] = 0.

    link_widths_orig = n.links.p_nom / linewidth_factor
    link_widths_orig[n.links.p_nom < line_lower_threshold] = 0.

    max_usage = n.links_t.p0.abs().max(axis=0)
    link_widths_used =  max_usage / linewidth_factor
    link_widths_used[max_usage < line_lower_threshold] = 0.

    link_color_used = n.links.carrier.map({"gas pipeline": "#f08080",
                                           "gas pipeline new": "#c46868"})

    n.links.bus0 = n.links.bus0.str.replace(" gas", "")
    n.links.bus1 = n.links.bus1.str.replace(" gas", "")

    tech_colors = snakemake.config['plotting']['tech_colors']

    bus_colors = {
        "fossil gas": tech_colors["fossil gas"],
        "methanation": tech_colors["methanation"],
        "biogas": "seagreen"
    }

    fig, ax = plt.subplots(figsize=(7,6),
                           subplot_kw={"projection": ccrs.PlateCarree()})

    n.plot(
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        link_colors='lightgrey',
        link_widths=link_widths_orig,
        branch_components=["Link"],
        ax=ax,
        **map_opts
    )

    n.plot(
        geomap=False,
        ax=ax,
        bus_sizes=0.,
        link_colors='#e8d1d1',
        link_widths=link_widths_rem,
        branch_components=["Link"],
        **map_opts
    )

    n.plot(
        geomap=False,
        ax=ax,
        bus_sizes=0.,
        link_colors=link_color_used,
        link_widths=link_widths_used,
        branch_components=["Link"],
        **map_opts
    )

    handles = make_legend_circles_for(
        [10e6, 100e6],
        scale=bus_size_factor,
        facecolor='grey'
    )
    labels = ["{} TWh".format(s) for s in (10, 100)]

    l2 = ax.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(-0.03, 1.01),
        labelspacing=1.0,
        frameon=False,
        title='gas generation',
        handler_map=make_handler_map_to_scale_circles_as_in(ax)
    )

    ax.add_artist(l2)

    handles = []
    labels = []

    for s in (50, 10):
        handles.append(plt.Line2D([0], [0], color="grey", linewidth=s * 1e3 / linewidth_factor))
        labels.append("{} GW".format(s))

    l1_1 = ax.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(0.28, 1.01),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1.5,
        title='gas pipeline used capacity'
    )

    ax.add_artist(l1_1)

    fig.savefig(
        snakemake.output.map.replace("-costs-all","-ch4_network"),
        bbox_inches="tight"
    )



def plot_series(network, carrier="AC", name="test"):

    n = network.copy()
    assign_location(n)
    assign_carriers(n)

    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):
        n_port = 4 if c.name=='Link' else 2
        for i in range(n_port):
            supply = pd.concat((supply,
                                (-1) * c.pnl["p" + str(i)].loc[:,
                                                               c.df.index[c.df["bus" + str(i)].isin(buses)]].groupby(c.df.carrier,
                                                                                                                     axis=1).sum()),
                               axis=1)

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(buses)]
        supply = pd.concat((supply, ((c.pnl["p"].loc[:, comps]).multiply(
            c.df.loc[comps, "sign"])).groupby(c.df.carrier, axis=1).sum()), axis=1)

    supply = supply.groupby(rename_techs_tyndp, axis=1).sum()

    both = supply.columns[(supply < 0.).any() & (supply > 0.).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.] = 0.
    negative_supply[negative_supply > 0.] = 0.

    supply[both] = positive_supply

    suffix = " charging"

    negative_supply.columns = negative_supply.columns + suffix

    supply = pd.concat((supply, negative_supply), axis=1)

    # 14-21.2 for flaute
    # 19-26.1 for flaute

    start = "2013-02-19"
    stop = "2013-12-25"

    threshold = 10e3

    to_drop = supply.columns[(abs(supply) < threshold).all()]

    if len(to_drop) != 0:
        print("dropping", to_drop)
        supply.drop(columns=to_drop, inplace=True)

    supply.index.name = None

    supply = supply / 1e3

    supply.rename(columns={"electricity": "electric demand",
                           "heat": "heat demand"},
                  inplace=True)
    supply.columns = supply.columns.str.replace("residential ", "")
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(["electric demand",
                                "transmission lines",
                                "hydroelectricity",
                                "hydro reservoir",
                                "run of river",
                                "pumped hydro storage",
                                "CHP",
                                "onshore wind",
                                "offshore wind",
                                "solar PV",
                                "solar thermal",
                                "building retrofitting",
                                "ground heat pump",
                                "air heat pump",
                                "resistive heater",
                                "OCGT",
                                "gas boiler",
                                "gas",
                                "natural gas",
                                "methanation",
                                "hydrogen storage",
                                "battery storage",
                                "hot water storage"])

    new_columns = (preferred_order.intersection(supply.columns)
                   .append(supply.columns.difference(preferred_order)))

    supply =  supply.groupby(supply.columns, axis=1).sum()

    plt.rcParams.update({'font.size': 12})

    for year in supply.index.levels[0]:
        fig, ax = plt.subplots()
        fig.set_size_inches((8, 5))
        fig.suptitle(year)

        (supply.loc[year].loc[start:stop, new_columns]
         .plot(ax=ax, kind="area", stacked=True, linewidth=0.,
               color=[snakemake.config['plotting']['tech_colors'][i.replace(suffix, "")]
                      for i in new_columns]))

        handles, labels = ax.get_legend_handles_labels()

        handles.reverse()
        labels.reverse()

        new_handles = []
        new_labels = []

        for i, item in enumerate(labels):
            if "charging" not in item:
                new_handles.append(handles[i])
                new_labels.append(labels[i])

        ax.legend(new_handles, new_labels, ncol=4, loc="upper left",
                  frameon=False, prop={'size': 10})
        ax.set_xlim([start, stop])
        ax.set_ylim([-2500, 2500])
        ax.grid(True)
        ax.set_ylabel("Power [GW]")
        fig.tight_layout()

        fig.savefig(
            snakemake.output.supply[:-4] + "_{}-{}.pdf".format(carrier, year),
            transparent=True,
            bbox_inches="tight"
        )

#%%
if __name__ == "__main__":
    if 'snakemake' not in globals():
        import os
        # os.chdir("/home/lisa/Documents/learning_curve/scripts")
        os.chdir("/home/lisa/mnt/lisa/learning_curve/scripts")
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        # "results/split_regions/configs/"
        # with open('/home/lisa/Documents/learning_curve/config.yaml', encoding='utf8') as f:
        with open('/home/lisa/mnt/lisa/learning_curve/results/oneEU_3learn_newupperbound/configs/config.yaml', encoding='utf8') as f:
            snakemake.config = yaml.safe_load(f)
        #overwrite some options
        sector_opts="Co2L-73sn-notarget-1p7-learnH2xElectrolysisp0-learnsolarp0-learnonwindp0-learnoffwindp0"
        snakemake.input = Dict()
        snakemake.input['network'] = "results/" + snakemake.config['run'] +"/postnetworks/elec_s_EU_{}.nc".format(sector_opts)
        snakemake.output = Dict(

        map="results/" + snakemake.config['run'] +"/maps/elec_s_EU_{}-costs-all.pdf".format(sector_opts),
        supply="results/" + snakemake.config['run'] +"/maps/elec_s_EU_{}-supply.pdf".format(sector_opts),)
        # os.chdir("/home/lisa/Documents/learning_curve/")
        os.chdir("/home/lisa/mnt/lisa/learning_curve/")
#
    n = pypsa.Network(snakemake.input.network,
                      override_component_attrs=override_component_attrs)

    map_opts = snakemake.config['plotting']['map']
    del map_opts["p_nom"]
    del map_opts["figsize"]

    if not snakemake.config["one_node"]:
        plot_map(n,
            components=["Generator", "Link", "Store"],
            bus_size_factor=1.5e10,
        )

        plot_h2_map(n)

    plot_series(n, carrier="AC")
    plot_series(n, carrier="heat")

    #plot_series(n, carrier="AC", name=suffix)
    #plot_series(n, carrier="heat", name=suffix)
