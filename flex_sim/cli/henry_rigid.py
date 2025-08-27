

from ase.io import read
from ase.build import molecule

from flex_sim.widom_insertion import WidomInsertion
import sys

cifpath = sys.argv[1]
modelpath = sys.argv[2]
jobname = sys.argv[3]
# Load the structure and build the gas
structure = read(cifpath)
gas = molecule("H2O")

# Create the WidomInsertion object
temperature = 293.15  # [K]
trajectory = f"widom_{jobname}.traj"
logfile = f"widom_{jobname}.log"
widom_insertion = WidomInsertion(
    structure,
    gas=gas,
    temperature=temperature,
    trajectory=trajectory,
    logfile=logfile,
    device="cuda",
    model_path=modelpath,
    default_dtype="float64",
    dispersion=False,
)
result = widom_insertion.run(num_insertions=5000, random_seed=0, fold=2)
print(result)
