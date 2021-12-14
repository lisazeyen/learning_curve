#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:00:15 2021

@author: bw0928
"""
import logging
from _helpers import configure_logging

import os
import pypsa_learning as pypsa
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_summary import rename_techs
to_rgba = mpl.colors.colorConverter.to_rgba

import numpy as np

from six import iteritems

from distutils.version import LooseVersion
pd_version = LooseVersion(pd.__version__)
agg_group_kwargs = dict(numeric_only=False) if pd_version >= "1.3" else {}

idx = pd.IndexSlice

logger = logging.getLogger(__name__)

opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}
#%%
def plot_generator_capacities(n):
    caps = n.generators.p_nom_opt.groupby([n.generators.carrier, n.generators.build_year]).sum(**agg_group_kwargs).drop("load", errors="ignore")
    (caps/1e3).unstack().plot(kind="bar", grid=True, stacked=True)
    plt.ylabel("GW")
    plt.savefig(snakemake.output.generator_caps, bbox_inches="tight")


def get_cap_per_investment_period(n, c):
    """
    returns the installed capacities for each investment period and component
    depending on build year and lifetime

    n: pypsa network
    c: pypsa component (e.g. "Generator")
    cap_per_inv: pd.DataFrame(index=investment_period, columns=components)

    """
    df = n.df(c)
    sns = n.snapshots
    cap_per_inv = pd.DataFrame(np.repeat([df.loc[:,df.columns.str.contains("_nom_opt")].iloc[:,0]],
                                         len(sns.levels[0]), axis=0),
                               index=sns.levels[0], columns=df.index)
    # decomissioned set caps to zero
    decomissioned_i = cap_per_inv.apply(lambda x: (x.index>df.loc[x.name, ["build_year", "lifetime"]].sum()-1))
    cap_per_inv[decomissioned_i] = 0
    # before build year set caps to zero
    not_build_i = cap_per_inv.apply(lambda x: x.index<df.loc[x.name, "build_year"])
    cap_per_inv[not_build_i] = 0

    return cap_per_inv

def get_co2_emissions(n):

    co2_glcs = ["Budget", "primary_energy"]
    glcs = n.global_constraints.query('type in @co2_glcs' )

    if glcs.empty: return

    weightings = (n.snapshot_weightings
                  .mul(n.investment_period_weightings["time_weightings"]
                       .reindex(n.snapshots).fillna(method="bfill").fillna(1.), axis=0)
                  )


    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f'{carattr} != 0')[carattr]

        if emissions.empty: continue

        # generators
        gens = n.generators.query('carrier in @emissions.index')
        if not gens.empty:
            em_pu = gens.carrier.map(emissions)/gens.efficiency
            em_pu = weightings["generator_weightings"].to_frame('weightings') @\
                    em_pu.to_frame('weightings').T
            emitted = n.generators_t.p[gens.index].mul(em_pu)

        return emitted



def plot_cap_per_investment_period(n, c):
    caps = get_cap_per_investment_period(n, c)
    caps = caps.groupby(n.df(c).carrier, axis=1).sum(**agg_group_kwargs).drop("load", axis=1, errors="ignore")
    caps = caps.groupby(caps.columns.map(rename_techs), axis=1).sum(**agg_group_kwargs)
    if not caps.empty:
        (caps/1e6).plot(kind="bar", stacked=True, grid=True, title="installed capacities",
                        color=[snakemake.config['plotting']['tech_colors'][i] for i in caps.columns])
        plt.ylabel("installed capacity \n [TW]")
        plt.xlabel("investment period")
        plt.legend(bbox_to_anchor=(1,1))
        plt.savefig(snakemake.output.generator_caps_per_inv, bbox_inches="tight")

def plot_generation(n, c):
    snakemake.config['plotting']['tech_colors']["load"] = "maroon"
    snakemake.config['plotting']['tech_colors']["demand"] = "maroon"
    snakemake.config['plotting']['tech_colors']["Link losses"] = "red"
    snakemake.config['plotting']['tech_colors']["Line losses"] = "pink"
    tot = pd.concat([
           n.generators_t.p.groupby(n.generators.carrier, axis=1).sum(**agg_group_kwargs).mul(n.snapshot_weightings.generator_weightings, axis=0),
           n.storage_units_t.p.groupby(n.storage_units.carrier, axis=1).sum(**agg_group_kwargs).mul(n.snapshot_weightings.store_weightings, axis=0),
           n.stores_t.p.groupby(n.stores.carrier, axis=1).sum(**agg_group_kwargs).mul(n.snapshot_weightings.store_weightings, axis=0),
           -1 * n.loads_t.p_set.mul(n.snapshot_weightings.generator_weightings, axis=0).sum(axis=1).rename("demand"),
           -1 * pd.concat([n.links_t.p0, n.links_t.p1], axis=1).sum(axis=1).mul(n.snapshot_weightings.generator_weightings, axis=0).rename("Link losses"),
           -1 * pd.concat([n.lines_t.p0, n.lines_t.p1], axis=1).sum(axis=1).mul(n.snapshot_weightings.generator_weightings, axis=0).rename("Line losses")], axis=1)

    (tot.groupby(level=0).sum()/1e6).plot(kind="bar", stacked=True, grid=True, title="generation",
                        color=[snakemake.config['plotting']['tech_colors'][i] for i in tot.columns])
    plt.ylabel("generation [TWh]")
    plt.xlabel("investment period")
    plt.legend(bbox_to_anchor=(1,1))
    plt.savefig(snakemake.output.p_per_inv, bbox_inches="tight")


def plot_co2_emissions(co2_emissions):

    grouped = co2_emissions.groupby(level=0).sum().groupby(n.generators.carrier, axis=1).sum(**agg_group_kwargs)
    grouped.plot(title="CO2 emissions", grid=True, kind="area")
    plt.ylabel("CO2 emissions [Mt]")
    plt.xlabel("investment period")


def plot_map(n, ax=None, opts={}):
    """
    """
    if ax is None:
        ax = plt.gca()

    ## DATA
    line_colors = {'cur': "purple",
                   'exp': mpl.colors.rgb2hex(to_rgba("red", 0.7), True)}
    tech_colors = opts['tech_colors']
    map_boundaries = opts['map']['boundaries']

    for carrier in ['H2 electrolysis', 'H2 fuel cell']:
        n.add("Carrier",
              carrier,
              color="pink")

    # generator caps
    investments = n.snapshots.levels[0]
    assign_location(n)
    caps = {}
    for c in ["Generator", "Link", "Store", "Line"]:
        active = pd.concat(
            [
                get_active_assets(n, c, inv_p, n.snapshots).rename(inv_p)
                for inv_p in investments
            ],
            axis=1,
        ).astype(int)
        caps[c] = ((active.mul(n.df(c)[opt_name.get(c, "p") + "_nom_opt"], axis=0))
                   .groupby([n.df(c).carrier, n.df(c).country]).sum(**agg_group_kwargs))



    for year in investments:
        # bus_sizes = n.generators_t.p.sum().loc[n.generators.carrier == "load"].groupby(n.generators.bus).sum()
        bus_sizes = pd.concat([caps["Generator"][year].drop("load"),
                               caps["Link"][year]])
        line_widths_exp = dict(Line=caps["Line"].loc["AC", year],
                               Link=caps["Link"].loc["DC", year])
        line_widths_cur = dict(Line=n.lines.s_nom_min, Link=n.links.loc[n.links[n.links.carrier=="DC"].index, "p_nom_min"])

        bus_sizes.rename(lambda x: x.replace(" discharger", ""), level=1, inplace=True)
        bus_sizes.rename(lambda x: x.replace(" charger", ""), level=1, inplace=True)

        line_colors_with_alpha = \
        dict(Line=(line_widths_cur['Line'] / n.lines.s_nom > 1e-3)
            .map({True: line_colors['cur'], False: to_rgba(line_colors['cur'], 0.)}),
            Link=(line_widths_cur['Link'] / n.links.p_nom > 1e-3)
            .map({True: line_colors['cur'], False: to_rgba(line_colors['cur'], 0.)}))

        ## FORMAT
        linewidth_factor = opts['map']['p_nom']['linewidth_factor']
        bus_size_factor  = opts['map']['p_nom']['bus_size_factor']

        ## PLOT
        n.plot(line_widths=pd.concat(line_widths_exp)/linewidth_factor,
               line_colors=dict(Line=line_colors['exp'], Link=line_colors['exp']),
               bus_sizes=bus_sizes/bus_size_factor,
               bus_colors=tech_colors,
               boundaries=map_boundaries,
               geomap=True,
               ax=ax)
        n.plot(line_widths=pd.concat(line_widths_cur)/linewidth_factor,
               line_colors=pd.concat(line_colors_with_alpha),
               bus_sizes=0,
               bus_colors=tech_colors,
               boundaries=map_boundaries,
               geomap=False,
               ax=ax)
        ax.set_aspect('equal')
        ax.axis('off')

        # Rasterize basemap
        # TODO : Check if this also works with cartopy
        for c in ax.collections[:2]: c.set_rasterized(True)

        # LEGEND
        handles = []
        labels = []

        for s in (10, 1):
            handles.append(plt.Line2D([0],[0],color=line_colors['exp'],
                                    linewidth=s*1e3/linewidth_factor))
            labels.append("{} GW".format(s))
        l1_1 = ax.legend(handles, labels,
                         loc="upper left", bbox_to_anchor=(0.24, 1.01),
                         frameon=False,
                         labelspacing=0.8, handletextpad=1.5,
                         title='Transmission Exist./Exp.             ')
        ax.add_artist(l1_1)

        handles = []
        labels = []
        for s in (10, 5):
            handles.append(plt.Line2D([0],[0],color=line_colors['cur'],
                                    linewidth=s*1e3/linewidth_factor))
            labels.append("/")
        l1_2 = ax.legend(handles, labels,
                    loc="upper left", bbox_to_anchor=(0.26, 1.01),
                    frameon=False,
                    labelspacing=0.8, handletextpad=0.5,
                    title=' ')
        ax.add_artist(l1_2)

        handles = make_legend_circles_for([10e3, 5e3, 1e3], scale=bus_size_factor, facecolor="w")
        labels = ["{} GW".format(s) for s in (10, 5, 3)]
        l2 = ax.legend(handles, labels,
                    loc="upper left", bbox_to_anchor=(0.01, 1.01),
                    frameon=False, labelspacing=1.0,
                    title='Generation',
                    handler_map=make_handler_map_to_scale_circles_as_in(ax))
        ax.add_artist(l2)

        techs =  (bus_sizes.index.levels[1]).intersection(pd.Index(opts['vre_techs'] + opts['conv_techs'] + opts['storage_techs']))
        handles = []
        labels = []
        for t in techs:
            handles.append(plt.Line2D([0], [0], color=tech_colors[t], marker='o', markersize=8, linewidth=0))
            labels.append(opts['nice_names'].get(t, t))
        l3 = ax.legend(handles, labels, loc="upper center",  bbox_to_anchor=(0.5, -0.), # bbox_to_anchor=(0.72, -0.05),
                    handletextpad=0., columnspacing=0.5, ncol=4, title='Technology')
#%%
if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('make_summary',
                           clusters='37', lv='1.0', sector_opts='Co2L-1H-73sn')

    n = pypsa.Network(snakemake.input.networks)
    plot_generator_capacities(n)

    plot_cap_per_investment_period(n, "Generator")

    plot_generation(n, "Generator")

    co2_emissions = get_co2_emissions(n)

    plot_co2_emissions(co2_emissions)

#%%
tot.columns = pd.MultiIndex.from_product([["solar learning"], tot.columns])
tot_no_learn.columns = pd.MultiIndex.from_product([["no learning"], tot_no_learn.columns])
a=pd.concat([tot, tot_no_learn], axis=1)
#%%
b = (a/1e6).stack(0)
b.plot(kind="bar", stacked=True, grid=True, title="generation - LR solar 38%",
                        color=[snakemake.config['plotting']['tech_colors'][i] for i in b.columns])
plt.ylabel("generation [TWh]")
plt.xlabel("investment period")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig("/home/ws/bw0928/Dokumente/learning_curve/results/test-network/graphics/compare_learning.pdf", bbox_inches="tight")
