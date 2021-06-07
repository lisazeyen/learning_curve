#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:00:41 2021

aggregate PyPSA network to representative periods

@author: bw0928
"""
import pandas as pd

import logging
logger = logging.getLogger(__name__)

# check if module is installed
from importlib.util import find_spec
if find_spec('tsam') is None:
    raise ModuleNotFoundError("Optional dependency 'tsam' not found."
                              "Install via 'pip install tsam'")

import tsam.timeseriesaggregation as tsam

from pyomo.environ import Var, NonNegativeReals
from pypsa.opt import l_constraint



def aggregate_snapshots(n, n_periods=12, hours=24, normed=True, solver="glpk",
                        extremePeriodMethod="None", clusterMethod='hierarchical',
                        predefClusterOrder=None, overwrite_time_dfs=True):
    """
    this function aggregates the snapshots of the pypsa Network to a number of
    typical periods (n_periods) with a given length in hours (hours).
    The mapping from the original timeseries to the aggregated typical periods
    is saved in n.cluster

    Default is 12 typical days with hierachical clustering and without
    any extremePeriodMethod.

    Parameters
    ----------
    n :                 pypsa.Network
    n_periods:          int
                        number of typical periods
    hours:              int
                        hours per period
    extremePeriodMethod: {'None','append','new_cluster_center',
                           'replace_cluster_center'}, default: 'None'
                        Method how to integrate extreme periods into to the
                        typical period profiles.
                        None: No integration at all.
                        'append': append typical Periods to cluster centers
                        'new_cluster_center': add the extreme period as additional cluster
                             center. It is checked then for all Periods if they fit better
                            to the this new center or their original cluster center.
                        'replace_cluster_center': replaces the cluster center of the
                            cluster where the extreme period belongs to with the periodly
                            profile of the extreme period. (Worst case system design)
    clusterMethod:      {'averaging', 'k_medoids', 'k_means', 'hierarchical'},
                        default: 'hierachical'
    predefClusterOrder: list or array (default: None)
                        Instead of aggregating a time series, a predefined
                        grouping is taken which is given by this list.
    overwrite_time_dfs: Bool (default:True), if set to True, time-dependent
                        attributes in pypsa are replaced by typical periods from
                        tsam module


    Returns
    -------


    """
    # create pandas dataframe with all time-dependent data of the pypsa network
    timeseries_df = prepare_timeseries(n, normed)

    logger.info(("Aggregate snapshots to {} periods with {} hours using "
                 "cluster method: {}, extreme period method: {}"
                .format(n_periods, hours, clusterMethod, extremePeriodMethod)))

    # get typical periods
    map_snapshots_to_periods, new_snapshots, timeseries_clustered = aggregate_timeseries(timeseries_df, n_periods, hours, extremePeriodMethod,
                                                                normed, clusterMethod, solver, predefClusterOrder)

    # save map original snapshots to typical periods
    n.cluster = map_snapshots_to_periods

    # set new snapshots
    n.set_snapshots(new_snapshots.index)
    n.snapshot_weightings = n.snapshot_weightings.mul(new_snapshots.weightings, axis=0)

    # set time dependent data according to typical tsam time series
    # if overwrite_time_dfs:
    #     overwrite_time_dependent(n, timeseries_clustered)



def prepare_timeseries(n, normed):
    """
    returns time dependent data to determinate typcial timeseries
    """

    timeseries_df = pd.DataFrame(index=n.snapshots)
    for component in n.all_components:
        pnl = n.pnl(component)
        for key in pnl.keys():
            if not pnl[key].empty:
                timeseries_df = pd.concat([timeseries_df, pnl[key]], axis=1)

    if normed:
        timeseries_agg = timeseries_df / timeseries_df.max()
    else:
        timeseries_agg = timeseries_df

    return timeseries_agg



def aggregate_timeseries(timeseries_df, n_periods, hours, extremePeriodMethod,
                         normed, clusterMethod, solver, predefClusterOrder):

    """
    aggregate timeseries with tsam module to a typical number of periods
    (n_periods) with each a lenght of hours
    """
    aggregation = tsam.TimeSeriesAggregation(
                            timeseries_df,
                            noTypicalPeriods=n_periods,
                            extremePeriodMethod=extremePeriodMethod,
                            rescaleClusterPeriods=False,
                            hoursPerPeriod=hours,
                            clusterMethod=clusterMethod,
                            solver=solver,
                            predefClusterOrder=predefClusterOrder,
                            )


    clustered = aggregation.createTypicalPeriods()
    if normed:
        clustered = clustered.mul(timeseries_df.max())

    map_snapshots_to_periods = aggregation.indexMatching()
    map_snapshots_to_periods["day_of_year"] = map_snapshots_to_periods.index.day_of_year
    cluster_weights = aggregation.clusterPeriodNoOccur
    clusterCenterIndices= aggregation.clusterCenterIndices

    # pandas Day of year starts at 1, clusterCenterIndices at 0
    new_snapshots = map_snapshots_to_periods[(map_snapshots_to_periods
                                             .day_of_year-1).isin(clusterCenterIndices)]
    new_snapshots["weightings"] = new_snapshots["PeriodNum"].map(cluster_weights).astype(float)
    clustered.set_index(new_snapshots.index, inplace=True)

    # last hour of typical period
    last_hour = new_snapshots[new_snapshots["TimeStep"]==hours-1]
    # first hour
    first_hour = new_snapshots[new_snapshots["TimeStep"]==0]

    # add typical period name and last hour to mapping original snapshot-> typical
    map_snapshots_to_periods["RepresentativeDay"] = map_snapshots_to_periods["PeriodNum"].map(last_hour.set_index(["PeriodNum"])["day_of_year"].to_dict())
    map_snapshots_to_periods["last_hour_RepresentativeDay"] = map_snapshots_to_periods["PeriodNum"].map(last_hour.reset_index().set_index(["PeriodNum"])["name"].to_dict())
    map_snapshots_to_periods["first_hour_RepresentativeDay"] = map_snapshots_to_periods["PeriodNum"].map(first_hour.reset_index().set_index(["PeriodNum"])["name"].to_dict())

    return map_snapshots_to_periods, new_snapshots, clustered


def overwrite_time_dependent(n, df_t):
    """
    overwrite time dependent data of pypsa network according to typical time
    series of tsam module
    """
    for component in n.all_components:
            pnl = n.pnl(component)
            for key in pnl.keys():
                if not pnl[key].empty:
                    pnl[key] = df_t.reindex(columns= pnl[key].columns)


def temporal_aggregation_storage_constraints(n, snapshots):
    """
    defines storage constraints for temporal aggregated pypsa network, which
    are added as an extra_functionality to the lopf for pyomo=True
    according to

    [1] Kotzur et. al. "Time series aggregation for energy system design:
        Modeling seasonal storage" (2018)
        https://doi.org/10.1016/j.apenergy.2018.01.023

    """
    # check if lopf is formulated with pyomo and timeseries aggregated
    if not hasattr(n, "model") and hasattr(n, "cluster"):
        raise AttributeError("this function can only be used as an "
                             "extra_functionaliy in the lopf with pyomo=True "
                             "and if the timeseries are aggregated to typcial "
                             "periods")
    logger.info("Setting storage unit constraints for inter- and intra periods. "
                " Storage unit parameters ['p_set', 'state_of_charge_set'] are "
                " not considered within this formulation. ")

    sus = n.storage_units
    # typical /representative periods
    typical_periods = n.cluster.RepresentativeDay.unique()
    # length of typical period
    hours = len(n.cluster.TimeStep.unique())
    # total number of periods = (len(all snapshots) / len(typical period))
    all_periods_i = pd.RangeIndex(len(n.cluster) / hours)

    # create variable for intra period state of charge (soc_intra)
    n.model.state_of_charge_intra = Var(sus.index, n.snapshots)

     # create variable for intra period state of charge (soc_inter)
    n.model.state_of_charge_inter = Var(sus.index, all_periods_i,
                                        within=NonNegativeReals)

    # INTRA PERIOD CONSTRAINTS #############################################
    # ---------------------------------------------------------------------
    def soc_intra_period_constraint(n, snapshots):
        """
        define intra period state of charge (soc_intra) constraints according to
        Kotzur et. al. [1] equation (18)

        soc intra at the beginning of each typical period is set to 0
        for the other intra period time steps soc constraints are defined by

        soc(t+1) = soc(t) p_store(t) - p_dispatch(t)
        soc = previous_soc  + p_store - p_dispatch

        returns soc_intra: dict which includes constraints
                           with keys [storageUnit, snapshot]
                           and values [lhs, sense, rhs]
                           e.g. lhs = [(-1, pyomo.var1), (1, pyomo.var2)]
                                sense = "=="
                                rhs = 0.0
        """

        soc_intra = {}
        sense = "=="
        rhs = 0.0

        for su in sus.index:
            for i,sn in enumerate(snapshots):
                soc_intra[su, sn] =  [[], sense, rhs]
                soc_intra_var = n.model.state_of_charge_intra[su,sn]
                soc_intra[su,sn][0].append((-1, soc_intra_var))
                # intra soc at the begining of each period is set to 0
                if i % hours == 0:
                    continue
                # otherwise normal soc constraints
                else:
                    previous_soc_intra = n.model.state_of_charge_intra[su, snapshots[i-1]]
                    soc_intra[su,sn][0].append(((1-sus.at[su,"standing_loss"]),
                                                  previous_soc_intra))
                    soc_intra[su,sn][0].append((sus.at[su,"efficiency_store"],
                                                n.model.storage_p_store[su,sn]))
                    soc_intra[su,sn][0].append((-(1/sus.at[su,"efficiency_dispatch"]),
                                          n.model.storage_p_dispatch[su,sn]))
        return soc_intra
    #-------------------------------------------------------------------------

    soc_intra = soc_intra_period_constraint(n, snapshots)
    l_constraint(n.model, "intra_state_of_charge_constraint",
                 soc_intra,list(sus.index), snapshots)


    # INTER PERIOD CONSTRAINTS #############################################
    # ---------------------------------------------------------------------
    def soc_inter_period_constraint(n, all_periods_i):
        """
        define inter period state of charge (soc_inter) constraints according to
        Kotzur et. al. [1] equation (19)

        soc_inter is equal to the previous soc_inter plus
        the intra period soc_intra of one hour after the last hour of the
        representaive previous period

        returns soc_inter: dict which includes constraints
                   with keys [storageUnit, period]
                   and values [lhs, sense, rhs]
                   e.g. lhs = [(-1, pyomo.var1), (1, pyomo.var2)]
                        sense = "=="
                        rhs = 0.0
        """

        soc_inter = {}
        sense = "=="
        rhs = 0.0

        # first snapshot of typical periods
        # first_sn = n.cluster.groupby("day_of_year").first()["last_hour_RepresentativeDay"]

        for su in sus.index:
            # standing losses for inter and intra period states
            eff_stand_inter = (1 - n.storage_units.at[su, 'standing_loss'])**hours
            eff_stand_intra = (1 - n.storage_units.at[su, 'standing_loss'])
            # efficiencies
            eff_dispatch = n.storage_units.at[su, 'efficiency_dispatch']
            eff_store = n.storage_units.at[su, 'efficiency_store']


            for i, period in enumerate(all_periods_i):
                # elapsed time
                eh = hours
                soc_inter[su, period] =  [[], sense, rhs]
                soc_inter_var = n.model.state_of_charge_inter[su,period]
                # lhs = -soc(period)
                soc_inter[su,period][0].append((-1, soc_inter_var))

                # soc_inter in the first period should be equal to last
                if i == 0:
                    last_soc_inter = n.model.state_of_charge_inter[su, all_periods_i[-1]]
                    # lhs = -soc_inter(first period) + soc_inter(last period)
                    soc_inter[su,period][0].append((1, last_soc_inter))
                else:
                    # lhs = -soc_inter(period) + soc_inter(period-1)
                    previous_soc_inter = n.model.state_of_charge_inter[su, all_periods_i[i-1]]
                    soc_inter[su, period][0].append((eff_stand_inter, previous_soc_inter))
                    # lhs += eff_stand_intra  * soc_intra(last_hour)
                    last_hour = n.cluster[::hours].iloc[period]["last_hour_RepresentativeDay"]
                    soc_inter[su, period][0].append((eff_stand_intra, n.model.state_of_charge_intra[su, last_hour]))
                    # lhs += (-1/eff_dispatch) * p_dispatch(last_hour)
                    soc_inter[su, period][0].append((-1/eff_dispatch * eh, n.model.storage_p_dispatch[su, last_hour]))
                    # lhs += eff_store * p_store(last_hour)
                    soc_inter[su, period][0].append((eff_store * eh, n.model.storage_p_store[su, last_hour]))

        return soc_inter
    #-------------------------------------------------------------------------

    soc_inter = soc_inter_period_constraint(n, all_periods_i)
    l_constraint(n.model, "inter_state_of_charge_constraint",
                 soc_inter, list(sus.index), all_periods_i)


    # NEW DEFINTIION OF STATE OF CHARGE (SOC) #########################

    # (1) state of charge is the sum of soc_intra and soc_inter
    # delete old soc constraints
    n.model.del_component('state_of_charge')
    n.model.del_component('state_of_charge_index')
    n.model.del_component('state_of_charge_index_0')
    n.model.del_component('state_of_charge_index_1')

    n.model.del_component('state_of_charge_constraint')
    n.model.del_component('state_of_charge_constraint_index')
    n.model.del_component('state_of_charge_constraint_index_0')
    n.model.del_component('state_of_charge_constraint_index_1')

    # ---------------------------------------------------------------------
    def total_soc(n, snapshots):
        """
        define total state of charge (soc) as sum of intra period state of
        charge (soc_intra) and inter period state of charge (soc inter),
        constraints according to Kotzur et. al. [1] equation (20)

        returns soc: dict which includes constraints
                           with keys [storageUnit, snapshot]
                           and values [lhs, sense, rhs]
                           e.g. lhs = [(-1, pyomo.var1), (1, pyomo.var2)]
                                sense = "=="
                                rhs = 0.0
        """

        soc = {}
        sense = "=="
        rhs = 0.0


        for su in sus.index:
            for i,sn in enumerate(snapshots):
                # dictionary values
                soc[su, sn] =  [[], sense, rhs]
                period = n.cluster.loc[sn]["RepresentativeDay"]
                # variables
                soc_var = n.model.state_of_charge[su,sn]
                soc_inter_var = n.model.state_of_charge_inter[su,period]
                soc_intra_var = n.model.state_of_charge_intra[su,sn]
                # lhs = -soc_intra - soc_inter_var + soc_var
                soc[su,sn][0].append((-1, soc_intra_var))
                soc[su,sn][0].append((-1, soc_inter_var))
                soc[su,sn][0].append((1, soc_var))

        return soc
    #-------------------------------------------------------------------------
    # define new soc variable with lower bound >= 0
    n.model.state_of_charge = Var(list(sus.index), snapshots,
                                  domain=NonNegativeReals, bounds=(0,None))

    # define constraint soc = soc_inter + soc_intra
    soc = total_soc(n, snapshots)
    l_constraint(n.model, "state_of_charge_constraint",
                 soc, list(sus.index), snapshots)

    # ------------------------------------------------------------------------
    # (2) SOC limits --------------------------------------------------------
    # (2)(a) state of charge upper limit
    # delete old soc upper limit constraints
    n.model.del_component('state_of_charge_upper')
    n.model.del_component('state_of_charge_upper_index')
    n.model.del_component('state_of_charge_upper_index_0')
    n.model.del_component('state_of_charge_upper_index_1')

    # ---------------------------------------------------------------------
    def soc_upper(n, snapshots):
        """
        define upper limit for soc

        returns soc_upper: dict which includes constraints
                           with keys [storageUnit, snapshot]
                           and values [lhs, sense, rhs]
                           e.g. lhs = [(-1, pyomo.var1), (1, pyomo.var2)]
                                sense = "=="
                                rhs = 0.0
        """

        soc_upper = {}
        sense = "<="

        for su in sus.index:
            # upper limit defined by storage capacity and max_hours
            # extendable
            if n.storage_units.p_nom_extendable[su]:
                p_nom = n.model.storage_p_nom[su]
                rhs = 0
                lhs = [(-n.storage_units.at[su, 'max_hours'], p_nom)]
            # non-extendable -> fixed capacities
            else:
                p_nom = n.storage_units.p_nom[su]
                rhs = p_nom * n.storage_units.at[su, 'max_hours']
                lhs = []

            for i, sn in enumerate(snapshots):
                # dictionary values
                soc_upper[su, sn] =  [lhs, sense, rhs]
                period = n.cluster.loc[sn]["RepresentativeDay"]
                # variables
                weightings = n.snapshot_weightings[sn]
                soc_intra_var = n.model.state_of_charge_intra[su, sn]
                soc_inter_var = n.model.state_of_charge_inter[su, period]
                # lhs
                soc[su,sn][0].append((1, soc_intra_var))
                soc[su,sn][0].append((1, soc_inter_var))

        return soc_upper
    #-------------------------------------------------------------------------

    soc_upper = soc_upper(n, snapshots)
    l_constraint(n.model, "state_of_charge_upper",
                 soc_upper, list(sus.index), snapshots)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # (3) cyclic constraint --------------------------------------------------
    def soc_cyclic(n, snapshots):
        """
        defines cyclic constraint
        """
        cyclic_i = sus[sus.cyclic_state_of_charge].index

        soc_cyclic = {}
        sense = "=="
        rhs = 0.

        for su in cyclic_i:
            # standing losses for inter and intra period states
            eff_stand_intra = (1 - n.storage_units.at[su, 'standing_loss'])
            # efficiencies
            eff_dispatch = n.storage_units.at[su, 'efficiency_dispatch']
            eff_store = n.storage_units.at[su, 'efficiency_store']

            # first soc
            first_soc = n.model.state_of_charge[su, snapshots[0]]
            lhs = [(-1, first_soc)]
            # last soc
            last_soc = n.model.state_of_charge[su, snapshots[-1]]
            lhs.append([(eff_stand_intra, last_soc)])
            # p dispatch last hour
            lhs.append([(-1/eff_dispatch, n.model.storage_p_dispatch[su, snapshots[-1]])])
            # p store last hour
            lhs.append([(eff_store, n.model.storage_p_store[su, snapshots[-1]])])

            # dictionary values
            soc_cyclic[su] =  [lhs, sense, rhs]

        return soc_cyclic

    # -----------------------------------------------------------------------
    soc_cyclic = soc_cyclic(n, snapshots)
    # l_constraint(n.model, "cyclic_storage_constraint",
    #              soc_cyclic, list(sus[sus.cyclic_state_of_charge].index))



