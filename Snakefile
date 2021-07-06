
configfile: "config.yaml"

wildcard_constraints:
    lv="[a-z0-9\.]+",
    clusters="[0-9]+m?",
    sector_opts="[-+a-zA-Z0-9\.\s]*"

subworkflow pypsaeur:
    workdir: "../pypsa-eur"
    snakefile: "../pypsa-eur/Snakefile"
    configfile: "../pypsa-eur/config.yaml"


rule solve_all_networks:
    input:
        expand("results/" + config['run'] + "/postnetworks/elec_s_{clusters}_lv{lv}_{sector_opts}.nc",
               **config['scenario'])

rule solve_all_single_networks:
    input:
        expand("results/" + config['run'] + "/postnetworks/DE_{sector_opts}_{clusters}.nc",
               **config['scenario'])

rule solve_all_sec_networks:
    input:
        expand("results/" + config['run'] + "/postnetworks/elec_s_EU_{sector_opts}.nc",
               **config['scenario'])

# rule solve_all_sec_networks:
#     input:
#         expand("results/" + config['run'] + "/postnetworks/elec_s_{clusters}_lv{lv}_{sector_opts}_sec.nc",
#                **config['scenario'])


rule solve_network:
    input:
        network="results/" + "prenetworks/elec_s_{clusters}.nc",
        config="results/"+ config['run'] + '/configs/config.yaml',
        global_capacity="data/global_capacities.csv",
        local_capacity="data/local_capacities.csv",
        costs="data/costs/",
        busmap_s="data/busmap_elec_s.csv",
        busmap="data/busmap_elec_s_{clusters}.csv",
        profile_offwind_ac="data/profile_offwind-ac.nc",
        profile_offwind_dc="data/profile_offwind-dc.nc",
    output: "results/" + config['run'] + "/postnetworks/elec_s_{clusters}_lv{lv}_{sector_opts}.nc"
    shadow: "shallow"
    log:
        solver="results/" + config['run'] + "/logs/elec_s_{clusters}_lv{lv}_{sector_opts}_solver.log",
        python="results/" + config['run'] + "/logs/elec_s_{clusters}_lv{lv}_{sector_opts}_python.log",
        memory="results/" + config['run'] + "/logs/elec_s_{clusters}_lv{lv}_{sector_opts}_memory.log"
    benchmark: "results/"+ config['run'] + "/benchmarks/_network/elec_s_{clusters}_lv{lv}_{sector_opts}"
    threads: 16
    resources: mem_mb=40000
    # group: "solve" # with group, threads is ignored https://bitbucket.org/snakemake/snakemake/issues/971/group-job-description-does-not-contain
    script: "scripts/prepare_and_solve_learning.py"

rule solve_network_single_ct:
    input:
        network="results/" + "prenetworks/DE.nc",
        config="results/"+ config['run'] + '/configs/config.yaml',
        global_capacity="data/global_capacities.csv",
        local_capacity="data/local_capacities.csv",
        costs="data/costs/",
        busmap_s="data/busmap_elec_s.csv",
        busmap="data/busmap_elec_s_{clusters}.csv",
        profile_offwind_ac="data/profile_offwind-ac.nc",
        profile_offwind_dc="data/profile_offwind-dc.nc",
    output: "results/" + config['run'] + "/postnetworks/DE_{sector_opts}_{clusters}.nc"
    shadow: "shallow"
    log:
        solver="results/" + config['run'] + "/logs/DE_{sector_opts}_{clusters}_solver.log",
        python="results/" + config['run'] + "/logs/DE_{sector_opts}_{clusters}_python.log",
        memory="results/" + config['run'] + "/logs/DE_{sector_opts}_{clusters}_memory.log"
    benchmark: "results/"+ config['run'] + "/benchmarks/_network/DE_{sector_opts}_{clusters}"
    threads: 16
    resources: mem_mb=40000
    # group: "solve" # with group, threads is ignored https://bitbucket.org/snakemake/snakemake/issues/971/group-job-description-does-not-contain
    script: "scripts/prepare_and_solve_learning.py"

rule solve_sec_network:
    input:
        network="results/" + "prenetworks/elec_s_EU.nc",
        config="results/"+ config['run'] + '/configs/config.yaml',
        global_capacity="data/global_capacities.csv",
        local_capacity="data/local_capacities.csv",
        costs="data/costs/",
        busmap_s="data/busmap_elec_s.csv",
        busmap="data/busmap_elec_s_37.csv",
        profile_offwind_ac="data/profile_offwind-ac.nc",
        profile_offwind_dc="data/profile_offwind-dc.nc",
        transport="data/transport/transport.csv",
        nodal_transport_data="data/transport/nodal_transport_data.csv",
        avail_profile="data/transport/avail_profile.csv",
        dsm_profile="data/transport/dsm_profile.csv",
    output: "results/" + config['run'] + "/postnetworks/elec_s_EU_{sector_opts}.nc"
    shadow: "shallow"
    log:
        solver="results/" + config['run'] + "/logs/elec_s_EU_{sector_opts}_sec_solver.log",
        python="results/" + config['run'] + "/logs/elec_s_EU_{sector_opts}_sec_python.log",
        memory="results/" + config['run'] + "/logs/elec_s_EU_{sector_opts}_sec_memory.log"
    benchmark: "results/"+ config['run'] + "/benchmarks/_network/elec_s_EU_{sector_opts}_sec"
    threads: 4
    resources: mem_mb=30000
    # group: "solve" # with group, threads is ignored https://bitbucket.org/snakemake/snakemake/issues/971/group-job-description-does-not-contain
    script: "scripts/prepare_and_solve_learning_sec.py"

# rule make_summary:
#     input:
#         networks="results/" + config['run'] + "/postnetworks/elec_s_{clusters}_lv{lv}_{sector_opts}.nc",
#         #heat_demand_name='data/heating/daily_heat_demand.h5'
#     output:
#         generator_caps="results/"+ config['run'] + '/graphics/generator_capacity_{clusters}_lv{lv}_{sector_opts}.pdf',
#         generator_caps_per_inv="results/"+ config['run'] + '/graphics/generator_capacity_per_inv_{clusters}_lv{lv}_{sector_opts}.pdf',
#         p_per_inv="results/"+ config['run'] + '/graphics/p_per_inv_{clusters}_lv{lv}_{sector_opts}.pdf',
#     threads: 2
#     resources: mem_mb=10000
#     script:
#         'scripts/make_summary.py'
#
# rule make_summary2:
#     input:
#         networks=expand("results/" + config['run'] +"/postnetworks/DE_{sector_opts}_{clusters}.nc",
#                  **config['scenario']),
#         costs="data/costs/costs_2030.csv",
#         plots=expand("results/" + config['run'] +"/postnetworks/DE_{sector_opts}_{clusters}.nc",
#               **config['scenario'])
#         #heat_demand_name='data/heating/daily_heat_demand.h5'
#     output:
#         nodal_costs="results" + '/' + config['run'] + '/csvs/nodal_costs.csv',
#         nodal_capacities="results"  + '/' + config['run'] + '/csvs/nodal_capacities.csv',
#         nodal_cfs="results"  + '/' + config['run'] + '/csvs/nodal_cfs.csv',
#         cfs="results"  + '/' + config['run'] + '/csvs/cfs.csv',
#         costs="results"  + '/' + config['run'] + '/csvs/costs.csv',
#         capacities="results"  + '/' + config['run'] + '/csvs/capacities.csv',
#         curtailment="results"  + '/' + config['run'] + '/csvs/curtailment.csv',
#         capital_costs_learning="results"  + '/' + config['run'] + '/csvs/capital_costs_learning.csv',
#         #energy="results"  + '/' + config['run'] + '/csvs/energy.csv',
#         #supply="results"  + '/' + config['run'] + '/csvs/supply.csv',
#         supply_energy="results"  + '/' + config['run'] + '/csvs/supply_energy.csv',
#         prices="results"  + '/' + config['run'] + '/csvs/prices.csv',
#         weighted_prices="results"  + '/' + config['run'] + '/csvs/weighted_prices.csv',
#         #market_values="results"  + '/' + config['run'] + '/csvs/market_values.csv',
#         #price_statistics="results"  + '/' + config['run'] + '/csvs/price_statistics.csv',
#         metrics="results"  + '/' + config['run'] + '/csvs/metrics.csv',
#         co2_emissions="results"  + '/' + config['run'] + '/csvs/co2_emissions.csv',
#         cumulative_capacities="results"  + '/' + config['run'] + '/csvs/cumulative_capacities.csv',
#         learn_carriers="results"  + '/' + config['run'] + '/csvs/learn_carriers.csv',
#     threads: 2
#     resources: mem_mb=10000
#     script:
#         'scripts/make_summary2.py'

rule make_summary_sec:
    input:
        networks=expand("results/" + config['run'] +"/postnetworks/elec_s_EU_{sector_opts}.nc",
                 **config['scenario']),
        costs="data/costs/costs_2020.csv",
        plots=expand("results/" + config['run'] +"/postnetworks/elec_s_EU_{sector_opts}.nc",
              **config['scenario'])
        #heat_demand_name='data/heating/daily_heat_demand.h5'
    output:
        nodal_costs="results" + '/' + config['run'] + '/csvs/nodal_costs.csv',
        nodal_capacities="results"  + '/' + config['run'] + '/csvs/nodal_capacities.csv',
        nodal_cfs="results"  + '/' + config['run'] + '/csvs/nodal_cfs.csv',
        cfs="results"  + '/' + config['run'] + '/csvs/cfs.csv',
        costs="results"  + '/' + config['run'] + '/csvs/costs.csv',
        capacities="results"  + '/' + config['run'] + '/csvs/capacities.csv',
        curtailment="results"  + '/' + config['run'] + '/csvs/curtailment.csv',
        capital_costs_learning="results"  + '/' + config['run'] + '/csvs/capital_costs_learning.csv',
        #energy="results"  + '/' + config['run'] + '/csvs/energy.csv',
        #supply="results"  + '/' + config['run'] + '/csvs/supply.csv',
        supply_energy="results"  + '/' + config['run'] + '/csvs/supply_energy.csv',
        prices="results"  + '/' + config['run'] + '/csvs/prices.csv',
        #weighted_prices="results"  + '/' + config['run'] + '/csvs/weighted_prices.csv',
        #market_values="results"  + '/' + config['run'] + '/csvs/market_values.csv',
        #price_statistics="results"  + '/' + config['run'] + '/csvs/price_statistics.csv',
        metrics="results"  + '/' + config['run'] + '/csvs/metrics.csv',
        co2_emissions="results"  + '/' + config['run'] + '/csvs/co2_emissions.csv',
        cumulative_capacities="results"  + '/' + config['run'] + '/csvs/cumulative_capacities.csv',
        learn_carriers="results"  + '/' + config['run'] + '/csvs/learn_carriers.csv',
    threads: 2
    resources: mem_mb=10000
    script:
        'scripts/make_summary2.py'

rule plot_summary:
    input:
        costs_csv="results"  + '/' + config['run'] + '/csvs/costs.csv',
        costs="data/costs/",
        # energy="results"  + '/' + config['run'] + '/csvs/energy.csv',
        balances="results"  + '/' + config['run'] + '/csvs/supply_energy.csv',
        eea ="data/eea/UNFCCC_v23.csv",
        countries="results"  + '/' + config['run'] + '/csvs/nodal_capacities.csv',
        co2_emissions="results"  + '/' + config['run'] + '/csvs/co2_emissions.csv',
        capital_costs_learning="results"  + '/' + config['run'] + '/csvs/capital_costs_learning.csv',
        capacities="results"  + '/' + config['run'] + '/csvs/capacities.csv',
        cumulative_capacities="results"  + '/' + config['run'] + '/csvs/cumulative_capacities.csv',
        learn_carriers="results"  + '/' + config['run'] + '/csvs/learn_carriers.csv',
    output:
        costs1="results"  + '/' + config['run'] + '/graphs/costs.pdf',
        costs2="results"  + '/' + config['run'] + '/graphs/costs2.pdf',
        costs3="results"  + '/' + config['run'] + '/graphs/total_costs_per_year.pdf',
        # energy="results"  + '/' + config['run'] + '/graphs/energy.pdf',
        balances="results"  + '/' + config['run'] + '/graphs/balances-energy.pdf',
        co2_emissions="results"  + '/' + config['run'] + '/graphs/carbon_budget_plot.pdf',
        capacities="results"  + '/' + config['run'] + '/graphs/capacities.pdf',
        capital_costs_learning="results"  + '/' + config['run'] + '/graphs/capital_costs_learning.pdf',
        learning_cost_vs_curve="results"  + '/' + config['run'] + '/graphs/learning_cost_vs_curve/learning_cost.pdf',
    threads: 2
    resources: mem_mb=10000
    script:
        'scripts/plot_summary.py'

rule copy_config:
    output:
        config="results/" + config['run'] + '/configs/config.yaml'
    threads: 1
    resources: mem_mb=1000
    script:
        'scripts/copy_config.py'

rule prepare_sec_network:
    input:
        network="results/" + "prenetworks/elec_s_37_lv1.0__Co2L0-1H-T-H-B-I-solar+p3-dist1_2020.nc",
        config="results/"+ config['run'] + '/configs/config.yaml',
        global_capacity="data/global_capacities.csv",
    output: "results/" + "prenetworks/elec_s_EU.nc"
    threads: 4
    resources: mem_mb=3000
    # group: "solve" # with group, threads is ignored https://bitbucket.org/snakemake/snakemake/issues/971/group-job-description-does-not-contain
    script: "scripts/build_one_node_sector_coupling.py"
