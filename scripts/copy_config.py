
from shutil import copy

files = ["config.yaml",
         "Snakefile",
         "scripts/prepare_perfect_foresight.py",
         "scripts/set_opts_and_solve.py",
         ]

for f in files:
    copy(f,'results/' + snakemake.config['run'] + '/configs/')
