from ase.io import read
from ase.build import molecule
import os
from flex_sim.widom_insertion import WidomInsertion
import sys

cif_folder = sys.argv[1]


log_files = "./logs"
if not os.path.exists(log_files):
    os.makedirs(log_files)
modelpath = "./MACE_stagetwo.model"

# Load the structure and build the gas
cif_ls = os.listdir(cif_folder)
cif_ls = [cif for cif in cif_ls if cif.endswith(".cif")]
gas = molecule("H2O")
# Create the WidomInsertion object
temperature = 293.15  # [K]


for idx, cif in enumerate(cif_ls):
    cifname = os.path.basename(cif).replace(".cif", "")
    trajectory = f"{log_files}/widom_{cifname}.traj"
    logfile = f"{log_files}/widom_{cifname}.log"
    structure = read(os.path.join(cif_folder, cif))
    widom_insertion = WidomInsertion(
        structure,
        gas=gas,
        init_structure_optimize=False,
        init_gas_optimize=False,
        temperature=temperature,
        trajectory=trajectory,
        logfile=logfile,
        device="cuda",
        model_path=modelpath,
        default_dtype="float64",
        dispersion=False,
    )
    result = widom_insertion.run(num_insertions=10000, random_seed=0, fold=1)
    print(result)
