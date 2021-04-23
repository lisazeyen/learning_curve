#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:21:27 2021

@author: bw0928
"""
import os, sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

os.chdir("/home/ws/bw0928/Dokumente/PyPSA")
sys.path = [os.pardir] + sys.path
import pypsa
print(pypsa.__file__)

solver_name="gurobi"
solver_io="python"
skip_pre=False
extra_functionality=None
solver_logfile=None
solver_options={}
keep_files=False
formulation="angles"
ptdf_tolerance=0.
free_memory={}
extra_postprocessing=None
solver_dir=None
warmstart = False
store_basis = False

lookup = pd.read_csv('pypsa/variables.csv',
                    index_col=['component', 'variable'])

from pypsa.pf import (_as_snapshots, get_switchable_as_dense as get_as_dense)
from pypsa.descriptors import (get_bounds_pu, get_extendable_i, get_non_extendable_i,
                          expand_series, nominal_attrs, additional_linkports, Dict,
                          get_active_assets, get_switchable_as_iter)

from pypsa.linopt import (linexpr, write_bound, write_constraint, write_objective,
                     set_conref, set_varref, get_con, get_var, join_exprs,
                     run_and_read_cbc, run_and_read_gurobi, run_and_read_glpk,
                     run_and_read_cplex, run_and_read_xpress,
                     define_constraints, define_variables, define_binaries,
                     align_with_static_component)


from numpy import inf

import gc, time, os, re, shutil
from tempfile import mkstemp

import logging
logger = logging.getLogger(__name__)


# Functions / helpers ---------------------------------------------------------
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


def get_social_discount(t, r=0.01):
    return (1/(1+r)**t)

def get_investment_weighting(energy_weighting, r=0.01):
    """
    returns cost weightings depending on the the energy_weighting (pd.Series)
    and the social discountrate r
    """
    end = energy_weighting.cumsum()
    start = energy_weighting.cumsum().shift().fillna(0)
    return pd.concat([start,end], axis=1).apply(lambda x: sum([get_social_discount(t,r)
                                                               for t in range(int(x[0]), int(x[1]))]),
                                                axis=1)
#%%
n = pypsa.Network()


# ## How to set snapshots and investment periods
# First set some parameters
# years of investment
years = [2020, 2030, 2040, 2050]
investment = pd.DatetimeIndex(['{}-01-01 00:00'.format(year) for year in years])
# temporal resolution
freq = "2190"
# snapshots (format -> DatetimeIndex)
snapshots = pd.DatetimeIndex([])
snapshots = snapshots.append([(pd.date_range(start ='{}-01-01 00:00'.format(year),
                                               freq ='{}H'.format(freq),
                                               periods=8760/float(freq))) for year in years])


# (b) or as a pandas MultiIndex, this will also change the investment_periods
# to the first level of the pd.MultiIndex
investment_helper = investment.union(pd.Index([snapshots[-1] + pd.Timedelta(days=1)]))
map_dict = {years[period] :
            snapshots[(snapshots>=investment_helper[period]) &
                      (snapshots<investment_helper[period+1])]
            for period in range(len(investment))}

multiindex = pd.MultiIndex.from_tuples([(name, l) for name, levels in
                                        map_dict.items() for l in levels])


n.set_snapshots(multiindex)


r = 0.01 # social discountrate
# set energy weighting -> last year is weighted by 1
n.investment_period_weightings.loc[:, 'time_weightings'] = n.investment_period_weightings.index.to_series().diff().shift(-1).fillna(1)
# n.investment_period_weightings.loc[:, 'generator_weightings'] = n.investment_period_weightings.index.to_series().diff().shift(-1).fillna(1)
# set investment_weighting
n.investment_period_weightings.loc[:, "objective_weightings"] = get_investment_weighting(n.investment_period_weightings["time_weightings"], r)
print(n.investment_period_weightings)


n.add("Bus",
      "bus 0")

# add some generators
p_nom_max = pd.Series((np.random.uniform() for sn in range(len(n.snapshots))),
                  index=n.snapshots, name="solar 2020")

# renewable
n.add("Generator",
      "solar 2020",
       bus="bus 0",
       build_year=2020,
       lifetime=31,
       marginal_cost=0.1,
       capital_cost=100,
       p_max_pu=p_nom_max,
       carrier="solar",
       p_nom_max=500,
       p_nom_extendable=True)


# conventional
n.add("Generator",
      "conventional 2020",
      bus="bus 0",
      build_year=2020,
      lifetime=31,
      marginal_cost=0.1,
      capital_cost=10,
      carrier="OCGT",
      p_nom_extendable=True)

n.add("Carrier",
      "solar",
      learning_rate=0.4,
      global_capacity=633700)


# add a Load
load_var =  pd.Series((100*np.random.uniform() for sn in range(len(n.snapshots))),
                  index=n.snapshots, name="load")
load_fix = pd.Series([250 for sn in range(len(n.snapshots))],
                  index=n.snapshots, name="load")

#add a load at bus 2
n.add("Load",
      "load",
      bus="bus 0",
      p_set=load_fix)


# (a) add CO2 Budget constraint ----------------------------------------------
n.add("GlobalConstraint",
      "CO2Budget",
      type="Budget",
      carrier_attribute="co2_emissions", sense="<=",
      constant=1e7)

path = "/home/ws/bw0928/Dokumente/learning_curve/results/prenetwork/single_bus.nc"
# n.export_to_netcdf(path)
#%%  LEARNING FUNCTIONS

# TODO
def learning_consistency_check(n):
    """
    if there is technology learning check
    (1) for carriers with technology learning there should be generators which
    are extendable
    (2) for generators with learning, there should be an upper capacity expansion
    limit (p_nom_max)
    (3) check that capital cost of generators with same learning carrier have
    the same investment costs
    (4) if carrier has technology learning -> global capacity has to be > 0
    """


def cumulative_cost_curve(cumulative_capacity, learning_rate, c0, initial_capacity=1):
    """
    instead of using the learning curve (eq 1)

    (1) c = c0 / (cumulative_capacity/initial_capacity)**alpha

    to avoid non-linearity in the objective function.
    Instead the cumulative costs are used. These are equal to the
    integral of the learning curve

    (2) cum_cost = integral(c) = 1/(1-alpha) * (c0 * cumulative_capacity * ((cumulative_capacity/initial_capacity)**-alpha))

     Input:
        ---------
        cumulative_capacity            : installed capacity (cumulative experience)
        learning_rate                  : Learning rate = (1-progress_rate), cost
                                         reduction with every doubling of cumulative
                                         capacity
        initial_capacity               : initial capacity (global),
                                         start experience, constant
        c0                             : initial investment costs, constant

        Return:
        ---------
            cumulative investment costs
    """
    # calculate alpha
    alpha = math.log10(1 / (1-learning_rate)) / math.log10(2)

    cum_cost =  1/(1-alpha) * (c0 * cumulative_capacity * ((cumulative_capacity/initial_capacity)**-alpha))

    return cum_cost



def piecewise_linear(x, y, segments, learning_rate, c0, e0):
    """
    defines interpolation points of piecewise linearisation of the learning
    curve.

    The higher the number of segments, the more precise is the solution. But
    with the number of segments, also the number of binary variables and
    therefore the solution time increases.

    Linearisation follows Barreto [1] approach of a dynamic segment: line segments
    at the beginning of the learning are shorter to capture the steep part
    of the learning curve more precisely. The segment length than doubles for
    the following line segments.

    Other models use the following number of segments:
        In ERIS [2] 6 and global MARKAL 6-8, MARKAL-Europe 6-20 [3].
        Heuberger et. al. [4] seem to use 4? that is shown in figure 2(b)


    [1] Barreto (2001) https://doi.org/10.3929/ethz-a-004215893 start at section 3.6 p.63
    [2] (2004) http://pure.iiasa.ac.at/id/eprint/7435/1/IR-04-010.pdf
    [3]  p.10 (1999) https://www.researchgate.net/publication/246064301_Endogenous_Technological_Change_in_Energy_System_Models
    [4] Heuberger et. al. (2017) https://doi.org/10.1016/j.apenergy.2017.07.075



    Inputs:
    ---------
        x        : x-point learning curve (cumulative installed capacity)
        y        : y-point learning curve (investment costs)
        segments : number of line segments (type: int) for piecewise
                   linearisation
    Returns:
    ----------
        fit      : pd.Series(y_fit, index=x_fit)
    """
    total = len(x)
    factor = 0
    x_index = [x.index[0]]
    for s in range(segments-1):
        factor += 2**s
    initial_len = total / factor
    x_index += [int(initial_len)*2**s for s in range(segments-1)]
    x_index.append(x.index[-1])

    x_fit = x.loc[x_index]
    y_fit = experience_curve(x_fit.values, learning_rate, c0, e0)

    return pd.DataFrame(y_fit, index=x_fit.index.values)


def experience_curve(cumulative_capacity, learning_rate, c0, initial_capacity=1):
    """
    calculates the specific investment costs for the corresponding cumulative
    capacity

    equations:

        (1) c = c0 / (cumulative_capacity/initial_capacity)**alpha

        (2) learning_rate = 1 - 1/2**alpha



    Input
    ---------
    cumulative_capacity            : installed capacity (cumulative experience)
    learning_rate                  : Learning rate = (1-progress_rate), cost
                                     reduction with every doubling of cumulative
                                     capacity
    initial_capacity               : initial capacity (global),
                                     start experience, constant
    c0                             : initial investment costs, constant

    Return
    ---------
    investment costs according to cumulative capacity, learning rate, initial
    costs and capacity
    """

    # calculate alpha
    alpha = math.log10(1 / (1-learning_rate)) / math.log10(2)
    # get specific investment
    return c0 / (cumulative_capacity/initial_capacity)**alpha



def define_learning_binaries(n, snapshots, segments=5):
    """
    defines binary variabe for generator learning for each
    investment period, each generator with learning rate and each segment of
    the linear interpolation of the learning curve

    binary_learning = 1 if cumulative CAPEX of generator carrier in investment
                        period on line segment else
                    = 0

    """
    # get carriers with learning
    learn_i = n.carriers[n.carriers.learning_rate!=0].index
    if learn_i.empty: return

    # get all investment periods
    investments = snapshots.levels[0] if isinstance(snapshots, pd.MultiIndex) else [0.]
    # create index for all line segments of the linear interpolation
    segments_i = pd.Index(np.arange(segments))

    # multiindex for every learning tech and pipe segment
    multi_i = pd.MultiIndex.from_product([learn_i, segments_i])
    # define binary variable (index=investment, columns=[generator, segment])
    define_binaries(n, (investments, multi_i), 'Carrier', 'learning')


def define_learning_binary_constraint(n, snapshots):
    """
    constraint for every tech/carrier and investment period select only one
    line segment
    """
    c, attr = 'Carrier', 'learning'

    # get carriers with learning
    learn_i = n.carriers[n.carriers.learning_rate!=0].index
    if learn_i.empty: return

    # get learning binaries
    learning = get_var(n, c, attr)
    # sum of all binaries at one investment period for each snapshot
    lhs = linexpr((1, learning)).groupby(level=0, axis=1).sum()
    # define constraint to always select just on line segment
    define_constraints(n, lhs, '=', 1, 'Carrier', 'select_segment')


def x_position_learning_curve(n, snapshots, segments=5):
    """
    define constraints to identify x_position on the learning curve
    (x=cumulative capacity, y=investment costs)
    """

    c, attr = 'Carrier', 'learning'

    # get carriers with learning
    learn_i = n.carriers[n.carriers.learning_rate!=0].index
    if learn_i.empty: return

    # get learning binaries ------------------------------------------------
    learning = get_var(n, c, attr)

    # bounds for cumulative capacity --------------------------------------------------
    x_low = n.carriers.loc[learn_i, "global_capacity"]
    # TODO could be increase by fraction of considered region/world??
    x_high = 10 * x_low

    # check that upper bound is not infinity
    if any(x_high.isin([np.inf])):
        logger.error("p_nom_max of generators with technology learning is "
                     "infinity. "
                     "Please set a upper limit for the capacity extension.")


    # get interpolation points (number of points = line segments + 1)

    # (1) get x and y points on learning curve
    x = pd.DataFrame(np.linspace(x_low, x_high, 1000),
                             columns = x_low.index)
    y = pd.DataFrame(index=x.index, columns=x.columns)
    y_cum = pd.DataFrame(index=x.index, columns=x.columns)
    for carrier in x.columns:
        learning_rate = n.carriers.loc[carrier, "learning_rate"]
        e0 = n.carriers.loc[carrier, "global_capacity"]
        c0 = n.generators.groupby(n.generators.carrier).first().loc[carrier, "capital_cost"]
        y[carrier] = x[carrier].apply(lambda x: experience_curve(x, learning_rate, c0, e0))
        y_cum[carrier] = x[carrier].apply(lambda x: cumulative_cost_curve(x, learning_rate, c0, e0))

    # get progressive cumulative investment costs
    y_cum = y_cum - y_cum.iloc[0]


def learning(network, sns):
    """
    modify objective function to include technology learning for pyomo=False
    """
    investments = sns.levels[0] if isinstance(sns, pd.MultiIndex) else [0.]


    objective_w_investment = (n.investment_period_weightings["objective_weightings"]
                             .reindex(investments).fillna(1.))

    objective_weightings = (n.snapshot_weightings
                            .mul(n.investment_period_weightings["objective_weightings"]
                                 .reindex(sns)
                            .fillna(method="bfill").fillna(1.), axis=0))

    # constant for already done investment
    nom_attr = nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = get_extendable_i(n, c)
        active = pd.concat([get_active_assets(n,c,inv_p,sns).rename(inv_p)
                          for inv_p in investments], axis=1).astype(int).loc[ext_i]
        active_i = active.index
        constant += (n.df(c)[attr][active_i] @
                    (active.mul(n.df(c).capital_cost[active_i], axis=0)
                     .mul(objective_w_investment))).sum()
    object_const = write_bound(n, constant, constant)
    write_objective(n, linexpr((-1, object_const), as_pandas=False)[0])
    n.objective_constant = constant

    # marginal cost
    for c, attr in lookup.query('marginal_cost').index:
        cost = (get_as_dense(n, c, 'marginal_cost', sns)
                .loc[:, lambda ds: (ds != 0).all()]
                .mul(objective_weightings.loc[sns, "objective_weightings"], axis=0))
        if cost.empty: continue
        terms = linexpr((cost, get_var(n, c, attr).loc[sns, cost.columns]))
        write_objective(n, terms)

    # investment
    for c, attr in nominal_attrs.items():
        ext_i = get_extendable_i(n, c)
        cost = n.df(c)['capital_cost'][ext_i]
        if cost.empty: continue
        active = pd.concat([get_active_assets(n,c,inv_p,sns).rename(inv_p)
                          for inv_p in investments], axis=1).astype(int).loc[ext_i]

        caps = expand_series(get_var(n, c, attr).loc[cost.index], investments).loc[cost.index]
        cost_weighted = active.mul(cost, axis=0).mul(objective_w_investment)
        terms = linexpr((cost_weighted, caps))
        write_objective(n, terms)


#%%
n.lopf(pyomo=False, solver_name="gurobi", skip_objective=True,
       multi_investment_periods=True,
       extra_functionality=learning)

caps = get_cap_per_investment_period(n, "Generator")
if not caps.empty:
    ax = caps.plot(kind="bar", stacked=True, grid=True, title="installed capacities", width=5)
    plt.ylabel("installed capacity \n [MW]")
    plt.xlabel("investment period")
    plt.legend(bbox_to_anchor=(1,1))

total = pd.concat([
           n.generators_t.p,
           # n.storage_units_t.p,
           # n.stores_t.p,
           -1 * n.loads_t.p_set,
           # -1 * pd.concat([n.links_t.p0, n.links_t.p1], axis=1).sum(axis=1).rename("Link losses"),
           # -1 * pd.concat([n.lines_t.p0, n.lines_t.p1], axis=1).sum(axis=1).rename("Line losses")
           ],axis=1)
total = total.groupby(total.columns, axis=1).sum()
total.plot(kind="bar", stacked=True, grid=True, title="Demand and Generation per snapshot")
plt.ylabel("Demand and Generation")
plt.xlabel("snapshot")
plt.legend(bbox_to_anchor=(1,1))