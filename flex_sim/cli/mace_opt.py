from ase.io import read, write
from mace.calculators import MACECalculator
from ase.optimize import BFGS
from glob import glob
import argparse
import os

class XYZTrajectoryWriter:
    def __init__(self, atoms, filename="trajectory.xyz", interval=5):
        self.atoms = atoms
        self.filename = filename
        self.interval = interval
        self.counter = 0
        # Clear file at the start
        open(self.filename, 'w').close()

    def __call__(self):
        self.counter += 1
        if self.counter % self.interval == 0:
            write(self.filename, self.atoms, append=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MD simulation and optimization")
    parser.add_argument('--cif', type=str, required=True, help='Input CIF file path')
    parser.add_argument('--stepMax', type=int, default=300, help='Maximum number of optimization steps')
    parser.add_argument('--save_xyz', action='store_true', help='Save the optimization configurations for comparison')
    parser.add_argument('--interval', type=int, default=1, help='Interval for writing frames')
    parser.add_argument('--suffix', type=str, default=None, help='Folder suffix for md (write) and sample (read)')
    parser.add_argument('--modelpath', type=str, default='/capstor/scratch/cscs/lyutao/modelfiles/mof420/MACE_run-5555.model', help='Path to the MACE model')
    parser.add_argument('--cueq', action='store_true', help='Use cueq acceleration for MD simulation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run MACE on (e.g., "cpu" or "cuda")')
    args = parser.parse_args()
    
    device = args.device
    stepMax = args.stepMax
    interval = args.interval
    model_path = args.modelpath
    cueq_choice = args.cueq
    suffix = args.suffix
    incif_fullpath = args.cif
    parent_dir = os.path.dirname(incif_fullpath)
    cifid = os.path.basename(incif_fullpath).split('.')[0]
    xyzfile = os.path.join(parent_dir, f'{cifid}_opt.xyz')

    if not os.path.exists(xyzfile):
        raise FileNotFoundError(f"XYZ file {xyzfile} does not exist. Please check the path.")
    dest_folder = os.path.join(parent_dir, f'opt_{suffix}' if suffix else 'opt')

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    if cueq_choice:
        mace_calc = MACECalculator(model_paths=[model_path], device=device, cu_eq=True) 
    else:
        mace_calc = MACECalculator(model_paths=[model_path], device=device)

    atoms = read(xyzfile, format='extxyz', index=":")  # Read the last frame from the XYZ file
    num_to_geoopt = len(atoms)
    for i, atom in enumerate(atoms):
        atom.set_calculator(mace_calc)
        opt = BFGS(atom, logfile=os.path.join(dest_folder,f'opt_{i}.log'))
        #opt.attach(XYZTrajectoryWriter(atoms, filename=os.path.join(dest_folder,f'opt_{i}.xyz'), interval=interval))
        opt.run(fmax=0.01, steps=stepMax)
        write(os.path.join(dest_folder, f'opt_{i}.xyz'), atom)
    
    if args.save_xyz:
        structure_ls = []
        
        atoms = read(incif_fullpath, format='cif')
        structure_ls.append(atoms)
        
        for i in range(num_to_geoopt):
            atoms = read(os.path.join(dest_folder, f'opt_{i}.xyz'),index="-1")
            structure_ls.append(atoms)
        write(os.path.join(dest_folder, f'{cifid}_compare.xyz'), structure_ls, format='extxyz')
        print(f"Finish optimization and save the configurations for comparison to {os.path.join(dest_folder, f'{cifid}_compare.xyz')}")