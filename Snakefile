
configfile: "config.yaml"

wildcard_constraints:
    lv="[a-z0-9\.]+",
    clusters="[0-9]+m?",
    sector_opts="[-+a-zA-Z0-9\.\s]*"

subworkflow pypsaeur:
    workdir: "../pypsa-eur"
    snakefile: "../pypsa-eur/Snakefile"
    configfile: "../pypsa-eur/config.yaml"



rule solve_all_sec_networks:
    input:
        expand("results/" + config['run'] + "/postnetworks/elec_s_EU_{sector_opts}.nc",
               **config['scenario'])

rule plot_all_networks:
    input:
        expand("results/" + config['run'] +"/maps/elec_s_EU_{sector_opts}-costs-all.pdf",
               **config['scenario'])

rule plot_all_memories:
    input:
        expand("results/" + config['run'] +"/graphs/memory_elec_s_EU_{sector_opts}.pdf",
               **config['scenario'])

rule prepare_perfect_foresight:
    input:
        network=expand("results/prenetworks/" + "elec_s_37_lv1.0__Co2L0-1H-T-H-B-I-solar+p3-dist1_{investment_periods}.nc", **config['scenario']),
        brownfield_network = lambda w: ("results/prenetworks-brownfield/" + "elec_s_37_lv1.0__Co2L0-1H-T-H-B-I-solar+p3-dist1_{}.nc"
                                        .format(str(config['scenario']["investment_periods"][0]))),
        config="results/"+ config['run'] + '/configs/config.yaml',
        global_capacity="data/global_capacities.csv",
        local_capacity="data/local_capacities.csv",
        costs="data/costs/",
        p_max_pu="data/generators_p_max_pu.csv",
        generators_costs="data/generators_costs.csv",
        busmap_s="data/busmap_elec_s.csv",
        busmap="data/busmap_elec_s_37.csv",
        profile_offwind_ac="data/profile_offwind-ac.nc",
        profile_offwind_dc="data/profile_offwind-dc.nc",
    output: "results/" + config['run'] + "/prenetworks/elec_s_37_lv1.0__Co2L0-1H-T-H-B-I-solar+p3-dist1_brownfield_all_years.nc"
    threads: 2
    resources: mem_mb=10000
    script: "scripts/prepare_perfect_foresight.py"

rule set_opts_and_solve:
    input:
        network="results/" + config['run'] + "/prenetworks/elec_s_37_lv1.0__Co2L0-1H-T-H-B-I-solar+p3-dist1_brownfield_all_years.nc",
        config="results/"+ config['run'] + '/configs/config.yaml',
        global_capacity="data/global_capacities.csv",
        local_capacity="data/local_capacities.csv",
        transport="data/transport/transport.csv",
        transport_scenarios="data/transport/transport_scenarios.csv",
        nodal_transport_data="data/transport/nodal_transport_data.csv",
        biomass_potentials='data/biomass_potentials.csv',
        industrial_demand="data/industrial_demand.csv",
        nodal_energy_totals="data/nodal_energy_totals.csv",
        costs="data/costs/",
    output:
        network="results/" + config['run'] + "/postnetworks/elec_s_EU_{sector_opts}.nc",
        sols="results/" + config['run'] + "/sols/elec_s_EU_{sector_opts}.csv",
    shadow: "shallow"
    log:
        solver="results/" + config['run'] + "/logs/elec_s_EU_{sector_opts}_sec_solver.log",
        python="results/" + config['run'] + "/logs/elec_s_EU_{sector_opts}_sec_python.log",
        memory="results/" + config['run'] + "/logs/elec_s_EU_{sector_opts}_sec_memory.log"
    benchmark: "results/"+ config['run'] + "/benchmarks/_network/elec_s_EU_{sector_opts}_sec"
    threads: 14
    resources: mem_mb= 28000 # 30000
    # group: "solve" # with group, threads is ignored https://bitbucket.org/snakemake/snakemake/issues/971/group-job-description-does-not-contain
    script: "scripts/set_opts_and_solve.py"


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
        capital_cost="results"  + '/' + config['run'] + '/csvs/capital_cost.csv',
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
        eea ="data/eea/UNFCCC_v24.csv",
        countries="results"  + '/' + config['run'] + '/csvs/nodal_capacities.csv',
        co2_emissions="results"  + '/' + config['run'] + '/csvs/co2_emissions.csv',
        capital_costs_learning="results"  + '/' + config['run'] + '/csvs/capital_costs_learning.csv',
        capacities="results"  + '/' + config['run'] + '/csvs/capacities.csv',
        cumulative_capacities="results"  + '/' + config['run'] + '/csvs/cumulative_capacities.csv',
        learn_carriers="results"  + '/' + config['run'] + '/csvs/learn_carriers.csv',
        capital_cost="results"  + '/' + config['run'] + '/csvs/capital_cost.csv',
        sols=expand("results/" + config['run'] +"/sols/elec_s_EU_{sector_opts}.csv",
                 **config['scenario']),
    output:
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

rule plot_network:
    input:
        network="results/" + config['run'] +"/postnetworks/elec_s_EU_{sector_opts}.nc",
    output:
        map="results/" + config['run'] +"/maps/elec_s_EU_{sector_opts}-costs-all.pdf",
        supply="results/" + config['run'] +"/maps/elec_s_EU_{sector_opts}-supply.pdf",
    threads: 2
    resources: mem_mb=10000
    benchmark: "results/" + config['run'] +"/maps/elec_s_EU_{sector_opts}"
    script: "scripts/plot_network.py"

rule plot_memory:
    input:
        log="results/" + config['run'] + "/logs/elec_s_EU_{sector_opts}_sec_memory.log",
    output:
        memory_plot="results/" + config['run'] +"/graphs/memory_elec_s_EU_{sector_opts}.pdf"
    threads: 2
    resources: mem_mb=2000
    script: "scripts/plot_memory.py"
