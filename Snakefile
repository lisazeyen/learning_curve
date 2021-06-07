
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

# rule solve_all_sec_networks:
#     input:
#         expand("results/" + config['run'] + "/postnetworks/elec_s_{clusters}_lv{lv}_{sector_opts}_sec.nc",
#                **config['scenario'])


rule solve_network:
    input:
        network="results/" + "prenetworks/elec_s_{clusters}.nc",
        config="results/"+ config['run'] + '/configs/config.yaml',
        global_capacity="data/global_capacities.csv",
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

# rule solve_sec_network:
#     input:
#         network="results/" + "prenetworks/elec_s_{clusters}_sec.nc",
#         config="results/"+ config['run'] + '/configs/config.yaml',
#         global_capacity="data/global_capacities.csv"
#     output: "results/" + config['run'] + "/postnetworks/elec_s_{clusters}_lv{lv}_{sector_opts}.nc"
#     shadow: "shallow"
#     log:
#         solver="results/" + config['run'] + "/logs/elec_s_{clusters}_lv{lv}_{sector_opts}_sec_solver.log",
#         python="results/" + config['run'] + "/logs/elec_s_{clusters}_lv{lv}_{sector_opts}_sec_python.log",
#         memory="results/" + config['run'] + "/logs/elec_s_{clusters}_lv{lv}_{sector_opts}_sec_memory.log"
#     benchmark: "results/"+ config['run'] + "/benchmarks/_network/elec_s_{clusters}_lv{lv}_{sector_opts}_sec"
#     threads: 4
#     resources: mem_mb=30000
#     # group: "solve" # with group, threads is ignored https://bitbucket.org/snakemake/snakemake/issues/971/group-job-description-does-not-contain
#     script: "scripts/prepare_and_solve_learning_sec.py"

rule make_summary:
    input:
        networks="results/" + config['run'] + "/postnetworks/elec_s_{clusters}_lv{lv}_{sector_opts}.nc",
        #heat_demand_name='data/heating/daily_heat_demand.h5'
    output:
        generator_caps="results/"+ config['run'] + '/graphics/generator_capacity_{clusters}_lv{lv}_{sector_opts}.pdf',
        generator_caps_per_inv="results/"+ config['run'] + '/graphics/generator_capacity_per_inv_{clusters}_lv{lv}_{sector_opts}.pdf',
        p_per_inv="results/"+ config['run'] + '/graphics/p_per_inv_{clusters}_lv{lv}_{sector_opts}.pdf',
    threads: 2
    resources: mem_mb=10000
    script:
        'scripts/make_summary.py'

rule copy_config:
    output:
        config="results/"+ config['run'] + '/configs/config.yaml'
    threads: 1
    resources: mem_mb=1000
    script:
        'scripts/copy_config.py'
