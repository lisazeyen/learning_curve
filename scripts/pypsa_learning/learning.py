# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Module for technology learning with PyPSA.

Created on Thu Apr 22 16:21:27 2021

@author: bw0928
"""
import os
import pandas as pd
import numpy as np
import math

import pypsa_learning as pypsa


from pypsa_learning.pf import get_switchable_as_dense as get_as_dense
from pypsa_learning.descriptors import (
    get_extendable_i,
    expand_series,
    nominal_attrs,
    get_active_assets,
)

from pypsa_learning.linopt import (
    linexpr,
    write_bound,
    write_objective,
    get_var,
    define_constraints,
    define_variables,
    define_binaries,
    write_SOS2_constraint
)

from distutils.version import LooseVersion

pd_version = LooseVersion(pd.__version__)
agg_group_kwargs = dict(numeric_only=False) if pd_version >= "1.3" else {}

import logging

logger = logging.getLogger(__name__)

lookup = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "variables.csv"),
    index_col=["component", "variable"],
)
#%%
# Functions / helpers ---------------------------------------------------------
def get_cap_per_investment_period(n, c):
    """Get active capacity per investment period.

    Return the installed capacities for each investment period and component
    depending on build year and lifetime

    n: pypsa network
    c: pypsa component (e.g. "Generator")
    cap_per_inv: pd.DataFrame(index=investment_period, columns=components)

    """
    df = n.df(c)
    sns = n.snapshots
    cap_per_inv = pd.DataFrame(
        np.repeat(
            [df.loc[:, df.columns.str.contains("_nom_opt")].iloc[:, 0]],
            len(sns.levels[0]),
            axis=0,
        ),
        index=sns.levels[0],
        columns=df.index,
    )
    # decomissioned set caps to zero
    decomissioned_i = cap_per_inv.apply(
        lambda x: (x.index > df.loc[x.name, ["build_year", "lifetime"]].sum() - 1)
    )
    cap_per_inv[decomissioned_i] = 0
    # before build year set caps to zero
    not_build_i = cap_per_inv.apply(lambda x: x.index < df.loc[x.name, "build_year"])
    cap_per_inv[not_build_i] = 0

    return cap_per_inv


def get_social_discount(t, r=0.01):
    """Calculate social discount rate."""
    return 1 / (1 + r) ** t


def get_investment_weighting(energy_weighting, r=0.01):
    """Get cost weightings.

    Return cost weightings depending on the the energy_weighting (pd.Series)
    and the social discountrate r
    """
    end = energy_weighting.cumsum()
    start = energy_weighting.cumsum().shift().fillna(0)
    return pd.concat([start, end], axis=1).apply(
        lambda x: sum([get_social_discount(t, r) for t in range(int(x[0]), int(x[1]))]),
        axis=1,
    )


####### LEARNING FUNCTIONS ------------------------------------------------


def learning_consistency_check(n):
    """Consistency check for technology learning.

    Checks if:
        (i) there are any learning technology
        (ii) an upper and lower bound for the capacity is defined
        (iii) for learning carriers there are also assets with extendable
        capacity
    """
    # check if there are any technologies with learning
    learn_i = n.carriers[n.carriers.learning_rate != 0].index
    if learn_i.empty:
        raise ValueError("No carriers with technology learning defined")

    # check if there are nan values
    if (n.carriers.loc[learn_i].isna()).any(axis=None):
        raise ValueError(
            "Learning technologies have nan values, please check " "in n.carriers."
        )

    # check that lower bound is > 0
    x_low = n.carriers.loc[learn_i, "global_capacity"]
    if any(x_low < 1):
        raise ValueError(
            "technology learning needs an lower bound for the capacity "
            "which is larger than zero. Please set a lower "
            "limit for the capacity at "
            "n.carriers.global_capacity for all technologies with learning."
        )

    # check that upper bound is not zero or infinity
    x_high = n.carriers.loc[learn_i, "max_capacity"]
    if any(x_high.isin([0, np.inf])) or any(x_high < x_low):
        raise ValueError(
            "technology learning needs an upper bound for the capacity "
            "which is nonzero and not infinity. Please set a upper "
            "limit for the capacity extension at "
            "n.carriers.max_capacity for all technologies with learning."
        )

    # check that for technologies with learning there are also assets which are extendable
    carrier_found = []
    for c, attr in nominal_attrs.items():
        ext_i = get_extendable_i(n, c)
        if "carrier" not in n.df(c) or n.df(c).empty:
            continue
        learn_assets = n.df(c)[n.df(c)["carrier"].isin(learn_i)].index
        learn_assets = ext_i.intersection(
            n.df(c)[n.df(c)["carrier"].isin(learn_i)].index
        )
        if learn_assets.empty:
            continue
        carrier_found += n.df(c).loc[learn_assets, "carrier"].unique().tolist()
    if not learn_i.difference(carrier_found).empty:
        raise ValueError(
            "There have to be extendable assets for all technologies with learning."
            " Check the assets with the following carrier(s):\n "
            "- {} \n".format(*learn_i.difference(carrier_found))
        )

    learn_rate = n.carriers.loc[learn_i, "learning_rate"]
    info = "".join(
        "- {} with learning rate {}%\n ".format(tech, rate)
        for tech, rate in zip(learn_i, learn_rate * 100)
    )
    logger.info(
        "Technology learning assumed for the following carriers: \n"
        + info
        + " The capital cost for assets with these carriers are neglected"
        " instead the in n.carriers.initial_cost defined costs are "
        "assumed as starting point for the learning"
    )


def experience_curve(cumulative_capacity, learning_rate, c0, initial_capacity=1):
    """Define experience curve.

    Calculates the specific investment costs c for the corresponding cumulative
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
    investment costs c according to cumulative capacity, learning rate, initial
    costs and capacity
    """
    # calculate learning index alpha
    alpha = math.log2(1 / (1 - learning_rate))
    # get specific investment
    return c0 / (cumulative_capacity / initial_capacity) ** alpha


def cumulative_cost_curve(
    cumulative_capacity, learning_rate, c0, initial_capacity=1, with_previous_TC=True
):
    """Define cumulative cost depending on cumulative capacity.

    Using directly the learning curve (eq 1)

    (1) c = c0 / (cumulative_capacity/initial_capacity)**alpha

    in the objective function would create non-linearities.
    Instead of (eq 1) the cumulative costs (eq 2) are used. These are equal to
    the integral of the learning curve

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
    # calculate learning index alpha
    alpha = math.log2(1 / (1 - learning_rate))

    # cost at given cumulative capacity
    cost = experience_curve(cumulative_capacity, learning_rate, c0, initial_capacity)

    # account for previous investments
    if with_previous_TC:
        c0 = 0

    if alpha == 1:
        return (
            initial_capacity
            * c0
            * (math.log10(cumulative_capacity) - math.log10(initial_capacity))
        )

    return (1 / (1 - alpha)) * (cumulative_capacity * cost - initial_capacity * c0)


def get_cumulative_cap_from_cum_cost(cumulative_cost, learning_rate, c0, e0):
    """Calculate cumulative capacity from given cumulative costs.

     Input:
    ---------
        cumulative_cost                : cumulative cost invested into the technology
        learning_rate                  : Learning rate = (1-progress_rate), cost
                                         reduction with every doubling of cumulative
                                         capacity
        c0                             : initial investment costs, constant
        e0                             : initial capacity (global),
                                         start experience, constant


    Return:
    ---------
        cumulative capacity
    """
    if cumulative_cost == 0:
        return e0
    # calculate learning index alpha
    alpha = math.log2(1 / (1 - learning_rate))

    if alpha == 1:
        a = cumulative_cost / (e0 * c0) + math.log10(e0)
        return 10 ** a

    return ((cumulative_cost * (1 - alpha) + c0 * e0) / (c0 * e0 ** alpha)) ** (
        1 / (1 - alpha)
    )


def piecewise_linear(x, y, segments, carrier, mini_first=True):
    """Define interpolation points of piecewise-linearised learning curve.

    The higher the number of segments, the more precise is the solution. But
    with the number of segments, also the number of binary variables and
    therefore the solution time increases.

    Linearisation follows Barreto [1] approach of a dynamic segment: line segments
    at the beginning of the learning are shorter to capture the steep part
    of the learning curve more precisely. The cumulative cost increase (y-axis)
    than doubles with each line segment.

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
        fit      : pd.DataFrame(columns=pd.MultiIndex([carrier], [x_fit, y_fit]),
                                index=interpolation points = (line segments + 1))
    """

    factor = pd.Series(np.arange(segments), index=np.arange(segments), dtype=float)
    factor = factor.apply(lambda x: 2 ** x)
    # maximum cumulative cost increase
    y_max_increase = y.max() - y.min()
    if mini_first:
        initial_len = y_max_increase / factor[:-1].max()
        y_fit_data = (
            [y.min()]
            + [y.iloc[1]]
            + [y.min() + int(initial_len) * 2 ** s for s in range(segments - 2)]
            + [y.max()]
        )
    else:
        initial_len = y_max_increase / factor.max()
        y_fit_data = (
            [y.min()]
            + [y.min() + int(initial_len) * 2 ** s for s in range(segments - 1)]
            + [y.max()]
        )
    y_fit = pd.Series(y_fit_data, index=np.arange(segments + 1))
    index = y_fit.apply(lambda x: np.searchsorted(y, x, side="left"))
    x_fit = x.iloc[index]
    y_fit = y.iloc[index]
    fit = pd.concat([x_fit, y_fit], axis=1).reset_index(drop=True)
    fit.columns = pd.MultiIndex.from_product([[carrier], ["x_fit", "y_fit"]])

    return fit


def get_linear_interpolation_points(n, x_low, x_high, segments):
    """Define interpolation points of the cumulative investment curve.

    Get interpolation points (x and y position) of piece-wise linearisation of
    cumulative cost function.

    Inputs:
    ---------
        n             : pypsa Network
        x_low         : (pd.Series) lower bound cumulative installed capacity
        x_high        : (pd.Series) upper bound cumulative installed capacity
        segments      : (int)       number of line segments for piecewise

    Returns:
    ----------
        points        : (pd.DataFrame(columns=pd.MultiIndex([carrier], [x_fit, y_fit]),
                                index=interpolation points = (line segments + 1)))
                        interpolation/kink points of cumulative cost function

    """
    # (1) define capacity range
    x = pd.DataFrame(np.linspace(x_low, x_high, 1000), columns=x_low.index)
    # y postion on learning curve
    y = pd.DataFrame(index=x.index, columns=x.columns)
    # y position on cumulaitve cost function
    y_cum = pd.DataFrame(index=x.index, columns=x.columns)
    # interpolation points
    points = pd.DataFrame()

    # piece-wise linearisation for all learning technologies
    for carrier in x.columns:
        learning_rate = n.carriers.loc[carrier, "learning_rate"]
        e0 = n.carriers.loc[carrier, "global_capacity"]
        c0 = n.carriers.loc[carrier, "initial_cost"]
        # cost per unit from learning curve
        y[carrier] = x[carrier].apply(
            lambda x: experience_curve(x, learning_rate, c0, e0)
        )
        # cumulative costs
        y_cum[carrier] = x[carrier].apply(
            lambda x: cumulative_cost_curve(
                x, learning_rate, c0, e0, with_previous_TC=True
            )
        )

        # get interpolation points
        points = pd.concat(
            [points, piecewise_linear(x[carrier], y_cum[carrier], segments, carrier)],
            axis=1,
        )

    return points


def get_slope(points):
    """Return the slope of the line segments."""
    point_distance = (points.shift() - points).shift(-1).dropna(axis=0)

    return point_distance.xs("y_fit", level=1, axis=1) / point_distance.xs(
        "x_fit", level=1, axis=1
    )


def get_interception(points, slope):
    """Get interception point with cumulative cost (y) axis."""
    return (
        points.xs("y_fit", axis=1, level=1)
        - (slope * points.xs("x_fit", axis=1, level=1))
    ).dropna()


def define_bounds(points, col, bound_type, investments, segments):
    """Define lower and  upper bounds.

    Input:
        points      : interpolation points
        col         : must be in ["x_fit", "y_fit"]
        investments : investment periods
        bound_type  : must be in ["lower", "upper"]
    """
    bound = expand_series(points.stack(-2)[col], investments).T.swaplevel(axis=1)

    if bound_type == "lower":
        bound = bound.drop(segments, axis=1, level=1)
    elif bound_type == "upper":
        bound = bound.groupby(level=0, axis=1).shift(-1).dropna(axis=1)
    else:
        logger.error("boun_type has to be either 'lower' or 'upper'")

    return bound.sort_index(axis=1)


def replace_capital_cost_with_learning(n):
    """Set capital cost of assets with learning after the optimisation.

    Replaces after the optimisation for all assets with technology learning
    the initial capital cost by the investment costs resulting from the
    learning.
    """
    # all technologies wih learning
    learn_i = n.carriers[n.carriers.learning_rate != 0].index
    investments = (
        n.snapshots.levels[0] if isinstance(n.snapshots, pd.MultiIndex) else [0.0]
    )
    if learn_i.empty:
        return

    cumulative_cap = n.sols["Carrier"]["pnl"]["cumulative_capacity"]
    learning_rate = n.carriers.learning_rate
    initial_cost = n.carriers.initial_cost
    initial_capacity = n.carriers.global_capacity

    learning_cost = pd.DataFrame(index=investments, columns=learn_i)
    for carrier in learn_i:
        learning_cost[carrier] = (
            cumulative_cap[[carrier]]
            .apply(
                lambda x: experience_curve(
                    x,
                    learning_rate.loc[x.index],
                    initial_cost.loc[x.index],
                    initial_capacity.loc[x.index],
                ),
                axis=1,
            )
            .fillna(method="ffill")
            .groupby(level=0)
            .first(**agg_group_kwargs)
        )

    for c, attr in nominal_attrs.items():
        if "carrier" not in n.df(c):
            continue
        ext_i = get_extendable_i(n, c)
        learn_assets = n.df(c)[n.df(c)["carrier"].isin(learn_i)].index
        learn_assets = ext_i.intersection(
            n.df(c)[n.df(c)["carrier"].isin(learn_i)].index
        )
        if learn_assets.empty:
            continue
        n.df(c)["capital_cost_initial"] = n.df(c)["capital_cost"]
        n.df(c).loc[learn_assets, "capital_cost"] = (
            n.df(c)
            .loc[learn_assets]
            .apply(lambda x: learning_cost.loc[x.build_year, x.carrier], axis=1)
        )


# # ---------------------------------------------------------------------------
##############################################################################


def define_learning_binaries(n, snapshots, segments):
    """Define binaries for technology learning.

    Define binary variabe for technology learning for each
    investment period, each carrier with learning rate and each segment of
    the linear interpolation of the learning curve

    binary_learning = 1 if cumulative capacity of generator carrier in investment
                        period on line segment else
                    = 0
    Input:
    ------
        n          : pypsa network
        snapshots  : time steps considered for optimisation
        segments   : type(int) number of line segments for linear interpolation
    """
    # get carriers with learning
    learn_i = n.carriers[n.carriers.learning_rate != 0].index
    if learn_i.empty:
        return

    # get all investment periods
    investments = snapshots.levels[0] if isinstance(snapshots, pd.MultiIndex) else [0.0]
    # create index for all line segments of the linear interpolation
    segments_i = pd.Index(np.arange(segments))

    # multiindex for every learning tech and pipe segment
    multi_i = pd.MultiIndex.from_product([learn_i, segments_i])
    # define binary variable (index=investment, columns=[carrier, segment])
    define_binaries(n, (investments, multi_i), "Carrier", "learning")


def define_learning_binary_constraint(n, snapshots):
    """Define constraints for the learning binaries.

    Constraints for the binary variables:
        (1) for every tech/carrier and investment period select only one
        line segment
        (2) experience grows or stays constant
    """
    c, attr = "Carrier", "learning"

    # get carriers with learning
    learn_i = n.carriers[n.carriers.learning_rate != 0].index
    if learn_i.empty:
        return

    # get learning binaries
    learning = get_var(n, c, attr)
    # (1) sum over all line segments
    lhs = (
        linexpr((1, learning))
        .groupby(level=0, axis=1)
        .sum(**agg_group_kwargs)
        .reindex(columns=learn_i)
    )
    # define constraint to always select just on line segment
    define_constraints(n, lhs, "=", 1, "Carrier", "select_segment")

    # (2) experience must grow constraints (p.67 Barretto, eq 19)
    # experience tech at t+1 either remains at segment or moves further
    delta_sum = linexpr((1, learning)).cumsum(axis=1)
    next_delta_sum = linexpr((-1, learning)).cumsum(axis=1).shift(-1).dropna()
    # sum_P=1^i (lambda (P, t) - lambda (P, t+1)) >= 0
    lhs = delta_sum.iloc[:-1] + next_delta_sum
    define_constraints(n, lhs, ">=", 0, "Carrier", "delta_segment_lb")
    # sum_P=i^N (lambda (P, t) - lambda (P, t+1)) <= 0
    d_revert = learning.reindex(columns=learning.columns[::-1])
    delta_revert_sum = linexpr((1, d_revert)).cumsum(axis=1)
    next_delta_revert_sum = linexpr((-1, d_revert)).cumsum(axis=1).shift(-1).dropna()
    lhs = delta_revert_sum.iloc[:-1] + next_delta_revert_sum
    lhs = lhs.reindex(columns=learning.columns)
    define_constraints(n, lhs, "<=", 0, "Carrier", "delta_segment_ub")


def define_learning_constraint(n, snapshots):
    """Define constraints for the learning binaries.

    Constraints for the binary variables:
        (1) for every tech/carrier and investment period select only one
        line segment
        (2) experience grows or stays constant
    """
    c, attr = "Carrier", "learning"

    # get carriers with learning
    learn_i = n.carriers[n.carriers.learning_rate != 0].index
    if learn_i.empty:
        return

    # get learning binaries
    learning = get_var(n, c, attr)
    # (1) sum over all line segments
    lhs = (
        linexpr((1, learning))
        .groupby(level=0, axis=1)
        .sum(**agg_group_kwargs)
        .reindex(columns=learn_i)
    )
    # define constraint to always select just on line segment
    define_constraints(n, lhs, "=", 1, "Carrier", "select_segment")

    # logger.info("Grow constraints are on")
    # # (2) experience must grow constraints (p.67 Barretto, eq 19)
    # # experience tech at t+1 either remains at segment or moves further
    # delta_sum = linexpr((1, learning)).cumsum(axis=1)
    # next_delta_sum = linexpr((-1, learning)).cumsum(axis=1).shift(-1).dropna()
    # # sum_P=1^i (lambda (P, t) - lambda (P, t+1)) >= 0
    # lhs = delta_sum.iloc[:-1] + next_delta_sum
    # define_constraints(n, lhs, ">=", 0, "Carrier", "delta_segment_lb")
    # # sum_P=i^N (lambda (P, t) - lambda (P, t+1)) <= 0
    # d_revert = learning.reindex(columns=learning.columns[::-1])
    # delta_revert_sum = linexpr((1, d_revert)).cumsum(axis=1)
    # next_delta_revert_sum = linexpr((-1, d_revert)).cumsum(axis=1).shift(-1).dropna()
    # lhs = delta_revert_sum.iloc[:-1] + next_delta_revert_sum
    # lhs = lhs.reindex(columns=learning.columns)
    # define_constraints(n, lhs, "<=", 0, "Carrier", "delta_segment_ub")


def define_learning_variables(n, snapshots, segments):
    """Define binaries for technology learning.

    Define continuos variabe for technology learning for each
    investment period, each carrier with learning rate and each point of
    the linear interpolation of the learning curve

    learning        : continuos variable [0,1] which is per period and carrier
                      a special ordered set of type 2 (SOS2)
    Input:
    ------
        n          : pypsa network
        snapshots  : time steps considered for optimisation
        segments   : type(int) number of points for linear interpolation
    """
    # get carriers with learning
    learn_i = n.carriers[n.carriers.learning_rate != 0].index
    if learn_i.empty:
        return

    # get all investment periods
    investments = snapshots.levels[0] if isinstance(snapshots, pd.MultiIndex) else [0.0]
    # create index for all line segments of the linear interpolation
    segments_i = pd.Index(np.arange(segments+1))

    # multiindex for every learning tech and pipe segment
    multi_i = pd.MultiIndex.from_product([learn_i, segments_i])
    # define learning variable (index=investment, columns=[carrier, segment])
    learning = define_variables(n, 0, 1, "Carrier", "learning",
                                axes=[investments, multi_i])

    # add SOS2 (special ordered Set 2) constraint:
    # (i) sum of learning variables per investment period and carrier == 1
    # (ii) only 2 neighboring variables are allowed to be non-zero
    for carrier in learn_i:
        for year in investments:
            sos2 = write_SOS2_constraint(n, learning.loc[[year], [carrier]])


def get_shift_learning(learning):
    """Shift learning binaries temporally and sets first investment period
    to no learning."""
    no_learn_in_first = pd.Series(0, index=learning.columns)
    first_i = no_learn_in_first.xs(0, level=1, drop_level=False).index
    no_learn_in_first.loc[first_i] = 1
    return learning.shift().fillna(no_learn_in_first)


def define_x_position(
    n, x_low, x_high, investments, multi_i, learn_i, points, segments, time_delay
):
    """Define capacity for each line segment of the linear interpolation.

    Define variable for capacity at each line segment "xs"
    (x-postion in learning curve and cumulative cost curve) as well as
    the corresponding constraints.

    Input:
    ------
        n          : pypsa network
        x_low      : type(pd.Series) lower capacity limit for each learning carrier
        x_high      : type(pd.Series) upper capacity limit for each learning carrier
        points     : type(pd.DataFrame) interpolation points
        investments: type(pd.Index) investment periods
        segments   : type(int) number of line segments for linear interpolation
        multi_i    : type(pd.MultiIndex), [learning carrier, line egments]
        learn_i    : type(pd.Index) carriers with learning
    """
    c = "Carrier"
    # ---- DEFINE VARIABLE
    # define xs capacity variables for each line segment, technology, investment_period
    xs = define_variables(n, 0, x_high.max(), c, "xs", axes=[investments, multi_i])

    # -----DEFINE CONSTRAINTS
    # get learning binaries
    learning = get_var(n, c, "learning")

    # define lower and upper boundaries for cumulative capacity at each line segment
    # in heuberger et. al. p.6 eq (40,41)
    # in Barretto p.66 eq (17) lambda=xs, delta=learning
    x_lb = define_bounds(points, "x_fit", "lower", investments, segments).reindex(
        columns=xs.columns
    )
    if time_delay:
        # define new variable for time delay in learning ---------------------
        xs_shift = define_variables(
            n, 0, x_high.max(), c, "xs_shift", axes=[investments, multi_i]
        )
        # define difference called xs_shift_dff ----------------
        xs_shift_diff = define_variables(
            n, 0, x_high.max(), c, "xs_shift_diff", axes=[investments, multi_i]
        )

        # define relation xs <-> xs_shift ------------------------------------
        # no learning first investment period
        exclude_cols = xs_shift.xs(0, level=1, axis=1, drop_level=False).columns
        lhs = linexpr(
            (1, xs_shift.iloc[0][[col not in exclude_cols for col in xs_shift.columns]])
        )
        define_constraints(n, lhs, "==", 0, "Carrier", "xs_shift_delay")

        # xs_shift_diff same for first investment period
        lhs = linexpr((1, xs_shift.iloc[0]), (-1, xs_shift_diff.iloc[0]))
        define_constraints(n, lhs, "==", 0, "Carrier", "xs_shift_diff_relation")

        # ----------------------------------------------------------
        # sum_segments xs_shift(inv_p, carrier) = sum_segments xs(inv_p,carrier)
        lhs = (
            linexpr((1, xs_shift), (-1, xs.reindex(xs_shift.columns, axis=1)))
            .groupby(level=0, axis=1)
            .sum(**agg_group_kwargs)
        )
        define_constraints(n, lhs, "==", 0, "Carrier", "xs_shift_xs_relation")

        # xs_diff
        lhs = (
            linexpr(
                (1, xs_shift.iloc[1:]),
                (-1, xs_shift.shift().iloc[1:]),
                (-1, xs_shift_diff.reindex(columns=xs_shift.columns).iloc[1:]),
            )
            .groupby(level=0, axis=1)
            .sum()
        )
        define_constraints(n, lhs, "==", 0, "Carrier", "xs_shift_diff_relation")

        # define lower and upper bound for xs_shift -------------------------
        learning_shift = learning.shift().dropna().reindex(xs_shift.columns, axis=1)

        lhs = linexpr(
            (1, xs_shift.iloc[1:]),
            (-x_lb.iloc[1:].reindex(xs_shift.columns, axis=1), learning_shift),
        )

        define_constraints(n, lhs, ">=", 0, "Carrier", "xs_shift_lb")

        x_ub = define_bounds(points, "x_fit", "upper", investments, segments).reindex(
            columns=xs.columns
        )
        x_ub_shift = expand_series(x_high.reindex(multi_i,level=0), investments[1:]).T
        lhs = linexpr(
            (1, xs_shift.iloc[1:]),
            (-x_ub_shift.reindex(columns=learning_shift.columns), learning_shift),
        )

        define_constraints(n, lhs, "<=", 0, "Carrier", "xs_shift_ub")

        # define lower and upper bound for xs_shift_diff -------------------
        x_lb_shift = expand_series(x_high.reindex(multi_i,level=0), investments[1:]).T
        lhs = linexpr(
            (1, xs_shift.iloc[1:]),
            (-x_lb_shift.reindex(columns=learning_shift.columns), learning_shift),
        )

        define_constraints(n, lhs, "<=", 0, "Carrier", "xs_shift_lb")

        x_ub = define_bounds(points, "x_fit", "upper", investments, segments).reindex(
            columns=xs.columns
        )
        lhs = linexpr((1, xs_shift_diff.iloc[1:]), (-x_high.max(), learning_shift),)

        define_constraints(n, lhs, "<=", 0, "Carrier", "xs_shift_diff_ub")

    lhs = linexpr((1, xs), (-x_lb, learning))

    define_constraints(n, lhs, ">=", 0, "Carrier", "xs_lb")

    x_ub = define_bounds(points, "x_fit", "upper", investments, segments).reindex(
        columns=xs.columns
    )
    lhs = linexpr((1, xs), (-x_ub, learning))

    define_constraints(n, lhs, "<=", 0, "Carrier", "xs_ub")


def define_cumulative_capacity(n, x_low, x_high, investments, learn_i, points):
    """Define global cumulative capacity.

    Define variable and constraint for global cumulative capacity.

    Input:
    ------
        n          : pypsa network
        x_low      : type(pd.Series) lower capacity limit for each learning carrier,
                     today's installed global capacity
        x_high      : type(pd.Series) upper capacity limit for each learning carrier
        investments: type(pd.Index) investment periods
        learn_i    : type(pd.Index) carriers with learning
    """
    c = "Carrier"

    # define variable for cumulative capacity (index=investment, columns=carrier)
    cum_cap = define_variables(
        n, x_low, x_high, c, "cumulative_capacity", axes=[investments, learn_i]
    )
    # x position of the interpolation points for each investment period
    points_x_fit = expand_series(points.xs("x_fit", level=1, axis=1).unstack(),
                                 investments).T
    # learning variables
    learning = get_var(n, c, "learning")
    # sum over all line segments (lambda) = cumulative installed capacity
    lhs = (
        linexpr((points_x_fit.reindex(columns=learning.columns), learning))
        .groupby(level=0, axis=1)
        .sum(**agg_group_kwargs)
        .reindex(columns=cum_cap.columns)
    )

    lhs += linexpr((-1, cum_cap))
    define_constraints(n, lhs, "=", 0, "Carrier", "cum_cap_definition")


def define_capacity_per_period(n, investments, learn_i,  snapshots):
    """Define new installed capacity per investment period.

    Define variable 'cap_per_period' for new installed capacity per investment
    period and corresponding constraints.

    Input:
    ------
        n          : pypsa network
        snapshots  : time steps considered for optimisation
        investments: type(pd.Index) investment periods
        learn_i    : type(pd.Index) carriers with learning
    """
    c = "Carrier"

    # fraction of global installed capacity
    global_factor = expand_series(n.df(c).loc[learn_i, "global_factor"], investments).T
    # cumulative capacity
    cum_cap = get_var(n, c, "cumulative_capacity")

    # define variable for new installed capacity per period
    cap_per_period = define_variables(
        n, 0, np.inf, c, "cap_per_period", axes=[investments, learn_i]  # cap_upper,
    )

    ## (1) cumulative capacity = initial capacity + sum_t (new installed cap) -
    lhs = linexpr((1, cum_cap), (-1 / global_factor, cap_per_period))
    lhs.iloc[1:] += linexpr((-1, cum_cap.shift().dropna()))

    rhs = pd.DataFrame(0.0, index=investments, columns=learn_i)
    rhs.iloc[0] = n.carriers.global_capacity.loc[learn_i]

    define_constraints(n, lhs, "=", rhs, "Carrier", "cap_per_period_definition")

    ## (2) connect new capacity per period to nominal capacity per asset ----
    lhs = linexpr((-1, cap_per_period))
    for c, attr in nominal_attrs.items():
        ext_i = get_extendable_i(n, c)
        if "carrier" not in n.df(c) or n.df(c).empty:
            continue
        learn_assets = ext_i.intersection(
            n.df(c)[n.df(c)["carrier"].isin(learn_i)].index
        )
        if learn_assets.empty:
            continue

        active = (pd.concat([
                    get_active_assets(n, c, inv_p, snapshots).rename(inv_p)
                    for inv_p in investments
                ],axis=1).astype(int).loc[learn_assets])
        # new build assets in investment period
        new_build = (active.apply(lambda x: x.diff().fillna(x.iloc[0]), axis=1)
                     .replace(-1, 0))

        # nominal capacity for each asset
        caps = expand_series(get_var(n, c, attr).loc[learn_assets], investments)
        carriers = n.df(c).loc[learn_assets, "carrier"].unique()

        lhs[carriers] += (
            linexpr((new_build, caps))
            .groupby(n.df(c)["carrier"])
            .sum(**agg_group_kwargs)
            .T.reindex(columns=carriers)
        )
    define_constraints(n, lhs, "=", 0, "Carrier", "cap_per_asset")


def define_cumulative_cost(n, points, investments, segments, learn_i, time_delay):
    """Define global cumulative cost.

    Define variable and constraints for cumulative cost
    starting at zero (previous installed capacity is not priced)

    Input:
    ------
        n          : pypsa network
        points     : type(pd.DataFrame) interpolation points
        investments: type(pd.Index) investment periods
        segments   : type(int) number of line segments for linear interpolation
        learn_i    : type(pd.Index) carriers with learning
    """
    c = "Carrier"

    # define cumulative cost variable --------------------------------------
    y_lb = define_bounds(points, "y_fit", "lower", investments, segments)
    y_ub = define_bounds(points, "y_fit", "upper", investments, segments)

    cum_cost_min = (
        y_lb.groupby(level=0, axis=1).min(**agg_group_kwargs).reindex(learn_i, axis=1)
    )
    cum_cost_max = (
        y_ub.groupby(level=0, axis=1).max(**agg_group_kwargs).reindex(learn_i, axis=1)
    )
    # upper bound has to be largerthan cum_cost_max with time delay!
    cum_cost = define_variables(
        n, cum_cost_min, np.inf, c, "cumulative_cost", axes=[investments, learn_i]
    )

    # ---- define cumulative costs constraints -----------------------------
    # get slope of line segments = cost per unit (EUR/MW) at line segment
    slope = get_slope(points)
    check = pd.concat(
        [n.carriers.loc[slope.columns, "initial_cost"], slope.loc[0]], axis=1
    ).rename(columns={0: "slope"})
    logger.info("----------------------------------------------\n {} \n".format(check))

    y_intercept = get_interception(points, slope)
    y_intercept_t = (
        expand_series(y_intercept.stack(), investments).swaplevel().T.sort_index(axis=1)
    )
    # Variables ---
    # learning variables
    learning = get_var(n, c, "learning")
    # y values of interpolation points expand per investment period
    points_y_fit = expand_series(points.xs("y_fit", level=1, axis=1).unstack(),
                                 investments).T
    #  make sure that columns have same order
    y_intercept_t = y_intercept_t.reindex(learning.columns, axis=1)

    # define cumulative cost
    lhs = linexpr((-1, cum_cost))

    if time_delay:
        # learning at previous investment period
        learning_shift = learning.shift().dropna()
        # TC(t) = TC(t-1) + slope(t-1) * [cap(t) - cap(t-1)]
        #########################
        # define new variable x_diff for [cap(t)-cap(t-1)] ------------------
        x_diff= define_variables(n, 0, np.inf, c, "x_diff",
                                 axes=[investments, learning.columns])

        # no learning first investment period ------------------------------
        exclude_cols = x_diff.xs(0, level=1, axis=1, drop_level=False).columns
        lhs = linexpr(
            (1, x_diff.iloc[0][[col not in exclude_cols for col in x_diff.columns]])
        )
        define_constraints(n, lhs, "==", 0, "Carrier", "x_diff_delay")

        # ----------------------------------------------------------
        # sum_segments xs_diff(inv_p, carrier) = cap_per_period
        cap_per_period = get_var(n, c, "cap_per_period")
        global_factor = expand_series(n.df(c).loc[learn_i, "global_factor"], investments).T
        lhs = linexpr((1, x_diff)).groupby(level=0, axis=1).sum(**agg_group_kwargs)
        lhs += linexpr((-1/global_factor, cap_per_period)).reindex(columns=lhs.columns)

        define_constraints(n, lhs, "==", 0, "Carrier", "xs_shift_xs_relation")
        # -------------------------------------------------------
        x_max = (expand_series(n.carriers.loc[learn_i, "max_capacity"], investments[1:]).T
                 .reindex(columns=learning_shift.columns, level=0))
        lhs = linexpr((1, x_diff.iloc[1:]),(-x_max, learning_shift))

        define_constraints(n, lhs, "<=", 0, "Carrier", "x_diff_ub")
        #################################
        # TODO
        # slope is defined for each line segment
        # x_diff is defined for each interpolation point
        initial_cost = expand_series(n.carriers.loc[slope.columns, "initial_cost"].reindex(x_diff.columns, level=0), investments).T
        slope_diff = (expand_series(slope.unstack(), investments).T
                      .reindex_like(x_diff).groupby(level=0, axis=1).shift()
                       .fillna(initial_cost, axis=1))
                      # .fillna(method="bfill", axis=1))

        # define cumulative cost TC(t)
        lhs = linexpr((-1, cum_cost))
        # slope(t-1) * [cap(t) - cap(t-1)] = slope * x_diff
        lhs += (linexpr((slope_diff, x_diff))
                .groupby(level=0, axis=1)
                .sum(**agg_group_kwargs)
                .reindex(lhs.columns, axis=1))
        # cumulative cost at previous investment period TC(t-1)
        lhs.iloc[1:] += (
            linexpr((1, cum_cost.shift().dropna()))
            .reindex(lhs.columns, axis=1)
        )

        rhs = pd.DataFrame(0, index=lhs.index, columns=lhs.columns)
        rhs.iloc[0, :] = -points_y_fit.iloc[0].xs(0, level=1)

    else:

        lhs += (
            linexpr((points_y_fit.reindex(learning.columns, axis=1), learning))
            .groupby(level=0, axis=1)
            .sum(**agg_group_kwargs)
            .reindex(lhs.columns, axis=1)
        )
        rhs = 0

    define_constraints(n, lhs, "=", rhs, "Carrier", "cum_cost_definition")


def define_cost_per_period(n, points, investments, segments, learn_i):
    """Define investment costs per investment period.

    Define investment costs for each technology per invesment period.

    Input:
    ------
        n          : pypsa network
        points     : type(pd.DataFrame) interpolation points
        investments: type(pd.Index) investment periods
        segments   : type(int) number of line segments for linear interpolation
        learn_i    : type(pd.Index) carriers with learning
    """
    c = "Carrier"

    # fraction of global installed capacity
    global_factor = expand_series(n.df(c).loc[learn_i, "global_factor"], investments).T

    # bounds  --------------------------------------
    y_lb = define_bounds(points, "y_fit", "lower", investments, segments)

    # define variable for investment per period in technology ---------------
    inv = define_variables(n, 0, np.inf, c, "inv_per_period",
                           axes=[investments, learn_i])
    # cumulative cost
    cum_cost = get_var(n, c, "cumulative_cost")
    # inv = cumulative_cost(t) - cum_cost(t-1)
    lhs = linexpr((1, cum_cost), (-1 / global_factor, inv))
    lhs.iloc[1:] += linexpr((-1, cum_cost.shift().dropna()))

    rhs = pd.DataFrame(0.0, index=investments, columns=lhs.columns)
    rhs.iloc[0] = points.xs("y_fit", level=1, axis=1).loc[0].reindex(lhs.columns)

    define_constraints(n, lhs, "=", rhs, "Carrier", "inv_per_period")


def define_position_on_learning_curve(n, snapshots, segments, time_delay):
    """Define piece-wised linearised learning curve.

    Define constraints to identify x_position on the learning curve
    (x=cumulative capacity, y=investment costs) and corresponding investment
    costs. The learning curve is expressed by the cumulative investments,
    which are piecewise linearised with line segments.
    """
    # ############ INDEX ######################################################
    # get carriers with learning
    learn_i = n.carriers[n.carriers.learning_rate != 0].index
    # get all investment periods
    investments = snapshots.levels[0] if isinstance(snapshots, pd.MultiIndex) else [0.0]

    # ############ BOUNDS #####################################################
    # bounds for cumulative capacity -----------------------------------------
    x_low = n.carriers.loc[learn_i, "global_capacity"]
    x_high = n.carriers.loc[learn_i, "max_capacity"]

    # ######## PIECEWIESE LINEARISATION #######################################
    # get interpolation points (number of points = line segments + 1)
    points = get_linear_interpolation_points(n, x_low, x_high, segments)

    # ######## CAPACITY #######################################################
    # ------------------------------------------------------------------------
    # define cumulative capacity
    define_cumulative_capacity(n, x_low, x_high, investments, learn_i, points)
    # -------------------------------------------------------------------------
    # define new installed capacity per period
    define_capacity_per_period(n, investments, learn_i,  snapshots)

    # ######## CUMULATIVE COST ################################################
    # ------- define cumulative cost -----------------------------------------
    define_cumulative_cost(n, points, investments, segments, learn_i, time_delay)
    # -------------------------------------------------------------------------
    # define new investment per period
    define_cost_per_period(n, points, investments, segments, learn_i)


def define_learning_objective(n, sns):
    """Modify objective function to include technology learning for pyomo=False.

    The new objective function consists of
    (i) the 'regular' equations of all the expenses for the technologies
          without learning,
    plus
    (ii) the investment costs for the learning technologies (learni_i)
    """
    investments = sns.levels[0] if isinstance(sns, pd.MultiIndex) else [0.0]

    objective_w_investment = (
        n.investment_period_weightings["objective_weightings"]
        .reindex(investments)
        .fillna(1.0)
    )

    objective_weightings = n.snapshot_weightings.mul(
        n.investment_period_weightings["objective_weightings"]
        .reindex(sns, level=0)
        .fillna(method="bfill")
        .fillna(1.0),
        axis=0,
    )

    # (i) non-learning technologies -----------------------------------------
    # constant for already done investment
    nom_attr = nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = get_extendable_i(n, c)
        active = (
            pd.concat(
                [
                    get_active_assets(n, c, inv_p, sns).rename(inv_p)
                    for inv_p in investments
                ],
                axis=1,
            )
            .astype(int)
            .loc[ext_i]
        )
        active_i = active.index
        constant += (
            n.df(c)[attr][active_i]
            @ (
                active.mul(n.df(c).capital_cost[active_i], axis=0).mul(
                    objective_w_investment
                )
            )
        ).sum()
    object_const = write_bound(n, constant, constant)
    write_objective(n, linexpr((-1, object_const), as_pandas=False)[0])
    n.objective_constant = constant

    # marginal cost
    for c, attr in lookup.query("marginal_cost").index:
        cost = (
            get_as_dense(n, c, "marginal_cost", sns)
            .loc[:, lambda ds: (ds != 0).all()]
            .mul(objective_weightings.loc[sns, "objective_weightings"], axis=0)
        )
        if cost.empty:
            continue
        terms = linexpr((cost, get_var(n, c, attr).loc[sns, cost.columns]))
        write_objective(n, terms)

    # investment
    # get carriers with learning
    learn_i = n.carriers[n.carriers.learning_rate != 0].index

    for c, attr in nominal_attrs.items():
        ext_i = get_extendable_i(n, c)
        if "carrier" not in n.df(c) or n.df(c).empty:
            continue
        learn_assets = n.df(c)[n.df(c)["carrier"].isin(learn_i)].index
        learn_assets = ext_i.intersection(
            n.df(c)[n.df(c)["carrier"].isin(learn_i)].index
        )

        # assets without tecnology learning
        no_learn = ext_i.difference(learn_assets)
        cost = n.df(c)["capital_cost"][no_learn]
        if cost.empty:
            continue
        active = pd.concat(
            [
                get_active_assets(n, c, inv_p, sns).rename(inv_p)
                for inv_p in investments
            ],
            axis=1,
        ).astype(int)

        caps = expand_series(get_var(n, c, attr).loc[no_learn], investments).loc[
            no_learn
        ]
        cost_weighted = (
            active.loc[no_learn].mul(cost, axis=0).mul(objective_w_investment)
        )
        terms = linexpr((cost_weighted, caps))
        write_objective(n, terms)

        # (ii) assets with technology learning -------------------------------
        if learn_assets.empty:
            continue
        cost_learning = get_var(n, "Carrier", "inv_per_period")
        # learning on anualised investment costs, additional weight for active time
        lifetime_w = (
            active.groupby([n.df(c).carrier, n.df(c).build_year])
            .first()
            .reindex(learn_i, level=0)
        )
        weighting = (
            lifetime_w.mul(objective_w_investment).sum(axis=1).unstack()[investments].T
        )
        terms = linexpr((weighting, cost_learning[weighting.columns]))
        write_objective(n, terms)

        # (iii) costs without learning
        if "nolearning_cost" in n.df(c).columns:
            logger.info("Non learning costs for component {} are added to objective.".format(c))

            nolearn_cost = n.df(c).loc[learn_assets, "nolearning_cost"].fillna(0)
            caps = expand_series(
                get_var(n, c, attr).loc[learn_assets], investments
            ).loc[learn_assets]
            cost_weighted = (
                active.loc[learn_assets].mul(nolearn_cost, axis=0).mul(objective_w_investment)
            )
            terms = linexpr((cost_weighted, caps))
            write_objective(n, terms)


def add_learning(n, snapshots, segments=5, time_delay=False):
    """Add technology learning to the lopf by piecewise linerarisation.

    Input:
        segments : type(int) number of line segments for linear interpolation
    """
    # consistency check
    learning_consistency_check(n)
    # learning variables of type SOS2
    define_learning_variables(n, snapshots, segments)
    # define constraints for learning variable
    define_learning_constraint(n, snapshots)
    # define relation cost - cumulative installed capacity
    define_position_on_learning_curve(n, snapshots, segments, time_delay)
    # define objective function with technology learning
    define_learning_objective(n, snapshots)
