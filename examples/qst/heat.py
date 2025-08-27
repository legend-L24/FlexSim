from flex_sim.u_md import simpleMD, GeometryOptimization, MD
from flex_sim.u_selector import selector
from ase.io import read, write
import random
from collections import Counter
import os
import numpy as np
import pandas as pd
from glob import glob
import argparse
from mace.calculators import MACECalculator
import ast

# 假设你有 N 个模型，每个模型预测出一个 shape 为 (304, 3) 的力矩阵
# forces_list 是一个长度为 N 的 list，每个元素是一个 shape 为 (304, 3) 的 numpy array
# 例如: forces_list = [np.random.rand(304, 3) for _ in range(4)]
def compute_force_deviation(forces_list):
    forces_array = np.stack(forces_list, axis=0)  # shape: (N_models, 304, 3)
    
    # 均值: shape (304, 3)
    mean_forces = np.mean(forces_array, axis=0)
    
    # 每个模型与均值之间的差的平方范数: shape (N_models, 304)
    squared_diffs = np.sum((forces_array - mean_forces) ** 2, axis=2)
    #print("squared_diffs shape: ", squared_diffs.shape)
    # 对模型求平均后再开方: shape (304,)
    force_deviation = np.sqrt(np.mean(squared_diffs, axis=0))
    #print("force_deviation shape: ", force_deviation.shape)
    
    # 最大原子力偏差
    max_force_deviation = np.max(force_deviation)
    
    return force_deviation, max_force_deviation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MD simulation and optimization")
    parser.add_argument('--cif', type=str, required=True, help='Input CIF file path')
    parser.add_argument('--suffix', type=str, default=None, help='Folder suffix for md (write) and sample (read)')
    parser.add_argument('--run_md', action='store_true', help='Run MD simulation')
    parser.add_argument('--numbers_to_run', type=int, default=1, help='Number of configurations to run')
    parser.add_argument('--run_devi', action='store_true', help='Run model deviation evaluation')
    parser.add_argument('--md_steps', type=int, default=10000, help='Number of MD steps')
    parser.add_argument('--modelpath', type=str, default='/users/lyutao/project/mil120_221/mace-medium-density-agnesi-stress.model', help='Path to the MACE model')
    parser.add_argument('--modellist', type=str, default=None, help='Path to the MACE model list')
    parser.add_argument('--cueq', action='store_true', help='Use cueq acceleration for MD simulation')
    parser.add_argument('--temperature_list',type=ast.literal_eval, default=[300], help='List of temperatures for MD simulation')
    args = parser.parse_args()
    
    
    model_path = args.modelpath
    cueq_choice = args.cueq
    
    incif_fullpath = args.cif
    parent_dir = os.path.dirname(incif_fullpath)
    cifid = os.path.basename(incif_fullpath).split('.')[0]
    suffix = args.suffix
    species = set(read(incif_fullpath).get_chemical_symbols())
    
    if args.modellist is not None:
        with open(args.modellist, 'r') as f:
            modellist = [line.strip() for line in f.readlines()]
            modellist = [p for p in modellist if p.endswith('.model')]
        print(f" 🚀 [INFO]  Read models to evaluate modeldeviation: {modellist}")
    
    outxyz = incif_fullpath
    print(f" 🚀 [INFO] Using original CIF file: {outxyz}")
    
    # Run MD simulation for sampling
    run_md = args.run_md
    temp_list = args.temperature_list
    if run_md:
        if cueq_choice:
            mace_calc = MACECalculator(model_paths=[model_path], device="cuda", cu_eq=True)  
        else:
            mace_calc = MACECalculator(model_paths=[model_path], device="cuda")  
        
        md_files = []
        os.makedirs(os.path.join(parent_dir, f'md_{suffix}'), exist_ok=True)
        for k in temp_list:
            print(f" 🚀 [INFO] Running MD simulation at {k}K")
            init_conf = read(outxyz)
            
            outf = f"{parent_dir}/md_{suffix}/{cifid}_{k}K.xyz"
            md_files.append(outf)
            MD(init_conf, temp=k, calc=mace_calc, fname=outf, s=5, T=args.md_steps)
            print(f" 🚀 [INFO] MD simulation finished, output saved to {outf}")
    
        ## Geometry optimization if not necessay in active learning step
    run_devi = args.run_devi
    if run_devi:
        
        if not args.modellist:
            raise ValueError("Please provide a model list for deviation evaluation.")
        
        if cueq_choice:
            print(f" 🚀 [INFO]  Using cueq acceleration for MD simulation")
            mace_calc_committe = [
                MACECalculator(model_paths=[p], device='cuda', cu_eq=True) for p in modellist
            ]
        else:
            mace_calc_committe = [
                    MACECalculator(model_paths=[p], device='cuda') for p in modellist
                ]
        md_files = sorted(glob(f'{parent_dir}/md_{suffix}/*.xyz'))
        md_files = [f for f in md_files if os.path.getsize(f) > 0]
        
        for outf in md_files:
            print(f" 🚀 [INFO] Deviation evaluation for configurations from {outf}")
            
            outf_basename = os.path.basename(outf)
            csv_filename = outf_basename.replace('.xyz', '_deviation.csv')
            outcsv = f"{parent_dir}/{csv_filename}"
            
            
            atoms = read(outf, index=':')
            
            devi_ener = []
            devi_forces_max = []
            devi_forces = []
            for atom in atoms:
                energies = np.array([])
                forces = []
                for idx, calc in enumerate(mace_calc_committe):
                    atom.calc = calc
                    energies = np.append(energies, atom.get_potential_energy())
                    # I wish forces as a new dimension
                    forces.append(atom.calc.get_forces())
                    atom.calc = None
                energies_devi = np.std(energies)
                per_atom_deviation, max_dev = compute_force_deviation(forces)
                devi_ener.append(energies_devi)
                devi_forces_max.append(max_dev)
                devi_forces.append(per_atom_deviation)
            devidata = pd.DataFrame({
                'deviation_energy': devi_ener,
                'deviation_forces_max': devi_forces_max,
                'deviation_forces': devi_forces
            })
            devidata.to_csv(outcsv, index=False)
            print(f" 🚀 [INFO]  Deviation data saved to {outcsv}")
