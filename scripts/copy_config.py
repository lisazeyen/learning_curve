
from shutil import copy

files = ["config.yaml",
         "Snakefile",
         "scripts/prepare_and_solve_learning.py",
         ]

for f in files:
    copy(f,'results/' + snakemake.config['run'] + '/configs/')
