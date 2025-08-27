from flex_sim.u_md import simpleMD, GeometryOptimization, run_md_npt
from flex_sim.u_selector import selector, selector_tail
from ase.io import read, write
import random
from collections import Counter
import os
import numpy as np
import pandas as pd
from glob import glob
import argparse
from mace.calculators import MACECalculator


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
    parser.add_argument('--run_md_opt', action='store_true', help='Run MD simulation and geoopt optimization')
    parser.add_argument('--run_md_multi', action='store_true', help='Run MD simulation on multiple configurations')
    parser.add_argument('--numbers_to_run', type=int, default=1, help='Number of configurations to run')
    parser.add_argument('--run_devi', action='store_true', help='Run model deviation evaluation')
    parser.add_argument('--make_devi', action='store_true', help='Based on deviation evaluation, choose the samples')
    parser.add_argument('--md_steps', type=int, default=10000, help='Number of MD steps')
    parser.add_argument('--run_sample', action='store_true', help='Do sampling or not')
    parser.add_argument('--run_sample_gas', action='store_true', help='Do sampling or not')
    parser.add_argument('--sort_sample', action='store_true', help='Sort sampled frames or not')
    parser.add_argument('--num_to_geo_opt', type=int, default=100, help='Random seed for frame selection')
    parser.add_argument('--num_to_sample', type=int, default=300, help='Number of frames to sample')
    parser.add_argument('--sample_test', action='store_true', help='Do test sampling or not')
    parser.add_argument('--num_to_sample_test', type=int, default=200, help='Number of test frames to sample')
    parser.add_argument('--modelpath', type=str, default='/users/lyutao/project/mil120_221/mace-medium-density-agnesi-stress.model', help='Path to the MACE model')
    parser.add_argument('--modellist', type=str, default=None, help='Path to the MACE model list')
    parser.add_argument('--cueq', action='store_true', help='Use cueq acceleration for MD simulation')
    parser.add_argument('--autosample', action='store_true', help='Run in auto mode')
    parser.add_argument('--autosample_gas', action='store_true', help='Run in auto mode')
    parser.add_argument('--totalnumber', type=int, default=14000, help='Total number of frames to sample')
    parser.add_argument('--number_of_gas', type=int, default=3, help='Do sampling or not')
    parser.add_argument('--simulation_type', type=str, default='NVT', help='Type of simulation to run (NVT, NPT, etc.)')
    args = parser.parse_args()
    
    simulation_type = args.simulation_type
    if simulation_type not in ['NVT', 'NPT']:
        raise ValueError("Invalid simulation type. Choose from 'NVT', 'NPT'.")
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
    if run_md:
        if cueq_choice:
            mace_calc = MACECalculator(model_paths=[model_path], device="cuda", cu_eq=True)  
        else:
            mace_calc = MACECalculator(model_paths=[model_path], device="cuda")  
        
        md_files = []
        os.makedirs(os.path.join(parent_dir, f'md_{suffix}'), exist_ok=True)
        for k in [300]:
            print(f" 🚀 [INFO] Running MD simulation at {k}K")
            init_conf = read(outxyz)
            
            outf = f"{parent_dir}/md_{suffix}/{cifid}_{k}K.xyz"
            md_files.append(outf)
            if simulation_type == 'NVT':
                    fig = simpleMD(init_conf, temp=k, calc=mace_calc, fname=outf, s=10, T=args.md_steps)
                    fig.savefig(f"{outf.split('.')[0]}.png")
            elif simulation_type == 'NPT':
                run_md_npt(init_conf, calculator=mace_calc, temperature=k, filename=outf, npt_time_fs=args.md_steps, nvt_time_fs=1000)
    
    run_md_opt = args.run_md_opt
    if run_md_opt:
        if cueq_choice:
            mace_calc = MACECalculator(model_paths=[model_path], device="cuda", cu_eq=True)  
        else:
            mace_calc = MACECalculator(model_paths=[model_path], device="cuda")  
        
        md_files = []
        os.makedirs(os.path.join(parent_dir, f'md_{suffix}'), exist_ok=True)
        for k in [300, 500]:
            print(f" 🚀 [INFO] Running MD simulation at {k}K")
            init_conf = read(outxyz)
            
            outf = f"{parent_dir}/md_{suffix}/{cifid}_{k}K.xyz"
            md_files.append(outf)
            if simulation_type == 'NVT':
                    fig = simpleMD(init_conf, temp=k, calc=mace_calc, fname=outf, s=10, T=args.md_steps)
                    fig.savefig(f"{outf.split('.')[0]}.png")
            elif simulation_type == 'NPT':
                run_md_npt(init_conf, calculator=mace_calc, temperature=k, filename=outf, npt_time_fs=args.md_steps, nvt_time_fs=1000)
        md_frames = []
        
        num_geo_opt = args.num_to_geo_opt
        print(f" 🚀 [INFO]  {num_geo_opt} frames selected for geometry optimization. ")
        for f in md_files:
            traj = read(f, index=':')
            md_frames += traj
        frames_to_go, idx_go = selector(md_frames, number=num_geo_opt, sort=False, species=species, seed=6666)
        flgs = []
        opt_files = []
        os.makedirs(os.path.join(parent_dir, f'opt_{suffix}'), exist_ok=True)
        for i, frame in zip(idx_go, frames_to_go):
            outf = f"{parent_dir}/opt_{suffix}/opt_{i}.xyz"
            logf = f"{parent_dir}/opt_{suffix}/opt.log"
            flg = GeometryOptimization(frame, outf, logf, 5, 0.01, 200, calc=mace_calc)
            flgs.append(flg)
            opt_files.append(outf)
        print(f"Geometry optimization finished! {Counter(flgs)}")
    
    run_md_multi = args.run_md_multi
    numbers_to_run = args.numbers_to_run
    if run_md_multi:
        
        sort = args.sort_sample
        
        if cueq_choice:
            mace_calc = MACECalculator(model_paths=[model_path], device="cuda", cu_eq=True)  
        else:
            mace_calc = MACECalculator(model_paths=[model_path], device="cuda")
        
        os.makedirs(os.path.join(parent_dir, f'md_{suffix}'), exist_ok=True)
        sampledata = read(f"{parent_dir}/{cifid}_sampledata.xyz", index=':')
        if len(sampledata) <= numbers_to_run:
            print(f"Not enough configurations to run: {len(sampledata)} < {numbers_to_run}")
            init_configs = sampledata
        else:
            init_configs, idx = selector(sampledata, number=numbers_to_run, sort=sort, species=species, seed=42)
        md_files = []
        for k in [300, 500]:
            print(f" 🚀 [INFO] Running MD simulation at {k}K")
            for idx, init_conf in enumerate(init_configs):
                outf = f"{parent_dir}/md_{suffix}/{cifid}_{k}K_{idx}.xyz"
                md_files.append(outf)
                if simulation_type == 'NVT':
                    fig = simpleMD(init_conf, temp=k, calc=mace_calc, fname=outf, s=10, T=args.md_steps)
                    fig.savefig(f"{outf.split('.')[0]}.png")
                elif simulation_type == 'NPT':
                    run_md_npt(init_conf, calculator=mace_calc, temperature=k, filename=outf, npt_time_fs=args.md_steps, nvt_time_fs=1000)

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
    
    make_devi = args.make_devi
    if make_devi:
        csv_data = [file for file in os.listdir(parent_dir) if file.endswith('_deviation.csv')]
        if not csv_data:
            raise ValueError("No deviation data found. Please run deviation evaluation first.")
        Property_key = 'deviation_forces_max'
        
        count=0
        all_count = 0
        for filename in csv_data:
            dest_dir = os.path.join(parent_dir, f'devi_{suffix}')
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            print(f" 🚀 [INFO]  Read deviation data from {filename}")
            df = pd.read_csv(os.path.join(parent_dir, filename))
            devi_data = df[Property_key].values
  
            xyz_path = os.path.join(os.path.join(parent_dir, f'md_{suffix}'), filename.replace('_deviation.csv', '.xyz'))
            atoms = read(xyz_path, index=':')
            # set the threshold
            
            #threshold_low, threshold_high = set_threshold(devi_data[0])
            threshold_low, threshold_high = 0.05, 0.4
            #print(f'Processing {filename} with threshold: {threshold_low} - {threshold_high} eV/A')
            indexes = [idx for idx, value in enumerate(devi_data) if value > threshold_low and value < threshold_high]
            all_count += len(devi_data)
            count += len(indexes)
            
            print(f'Number of structures with deviation in range: {len(indexes)}')
            picked_structures = [atom for idx, atom in enumerate(atoms) if idx in indexes]
  
            write(os.path.join(dest_dir, filename.replace('_deviation.csv', '_devi.xyz')), picked_structures)
        print(f" 🚀 [INFO]  Selected {count} structures from {all_count}")
        print(f" 🚀 [INFO]  Maximum force deviation of {count/all_count} data is above 0.05")
        
    auto_sample_gas = args.autosample_gas
    totalnumber = args.totalnumber
    sort = args.sort_sample
    if auto_sample_gas:
        devi_files = sorted(glob(f'{parent_dir}/devi_{suffix}/*.xyz'))
        devi_files = [f for f in devi_files if os.path.getsize(f) > 0]       
        all_files = devi_files
        print(f" 🚀 [INFO]  Read files from: {all_files}")
        all_frames = []
        for f in all_files:
            traj = read(f, index=':')
            print(f" 🚀 [INFO]  select {len(traj)} frames from {f}")
            all_frames += traj
        print(f"{len(all_frames)} frames from deviation evaluation")
        
        ## Choose the number of frames to DFT label
        ratio = len(all_frames) / totalnumber
        print(f" 🚀 [INFO]  Ratio of frames with larger force deviation than 0.05 to total number: {ratio:.4f}")
        if ratio < 0.005 and len(all_frames) > 23:
            num_to_sample = 10
            num_to_sample_test = 3
        elif 0.005 <=ratio < 0.01 and len(all_frames) > 33:
            num_to_sample = 30
            num_to_sample_test = 3
        elif 0.01 <= ratio < 0.04 and len(all_frames) > 66:
            num_to_sample = 66
            num_to_sample_test = 6
        elif 0.04 <= ratio < 0.1 and len(all_frames) > 100:
            num_to_sample = 100
            num_to_sample_test = 10
        elif 0.1 <= ratio < 0.3 and len(all_frames) > 220:
            num_to_sample = 200
            num_to_sample_test = 20
        elif 0.3 <= ratio and len(all_frames) > 330:
            num_to_sample = 300
            num_to_sample_test = 30
        else:
            num_to_sample = int(len(all_frames) * 0.7)
            num_to_sample_test = int(len(all_frames) * 0.1)
        print(f" 🚀 [INFO]  Number of frames to sample: {num_to_sample}, test frames: {num_to_sample_test}")
        number_of_gas = args.number_of_gas
        if num_to_sample and num_to_sample_test:
            frames_train, idx = selector_tail(all_frames, number=num_to_sample, sort=sort, species=species, seed=42, n_tail=number_of_gas)
            s_outf = os.path.join(parent_dir, f'{cifid}_sample{num_to_sample}.xyz')
            write(s_outf, frames_train)
            print(f" 🚀 [INFO]  Train sample: {num_to_sample} frames from {len(all_frames)} total frames")
            print(f"Selected frames: saved to {s_outf}")
            print(f"Sorted frames saved to {s_outf}")

            idx_left = set(np.arange(len(all_frames))) - set(idx)       
            left_frames = []
            for idx, atom in enumerate(all_frames):
                if idx in idx_left:
                    left_frames.append(atom)
            print(f" 🚀 [INFO]  Test sample: {num_to_sample_test} test frames from left {len(idx_left)} frames")
            frames_test, _ = selector_tail(left_frames, number=num_to_sample_test, sort=sort, species=species, seed=42, n_tail=number_of_gas)
            t_outf = os.path.join(parent_dir, f'{cifid}_sampletest.xyz')
            write(t_outf, frames_test)
            
            print(f"Selected test frames: saved to {t_outf}")
            if sort:
                print(f"Sorted test frames saved to {t_outf}")
                
    auto_sample = args.autosample
    if auto_sample:
        devi_files = sorted(glob(f'{parent_dir}/devi_{suffix}/*.xyz'))
        devi_files = [f for f in devi_files if os.path.getsize(f) > 0]       
        all_files = devi_files
        print(f" 🚀 [INFO]  Read files from: {all_files}")
        all_frames = []
        for f in all_files:
            traj = read(f, index=':')
            print(f" 🚀 [INFO]  select {len(traj)} frames from {f}")
            all_frames += traj
        print(f"{len(all_frames)} frames from deviation evaluation")
        
        ## Choose the number of frames to DFT label
        ratio = len(all_frames) / totalnumber
        print(f" 🚀 [INFO]  Ratio of frames with larger force deviation than 0.05 to total number: {ratio:.4f}")
        if ratio < 0.005 and len(all_frames) > 13:
            num_to_sample = 10
            num_to_sample_test = 3
        elif 0.005 <=ratio < 0.01 and len(all_frames) > 33:
            num_to_sample = 30
            num_to_sample_test = 3
        elif 0.01 <= ratio < 0.04 and len(all_frames) > 66:
            num_to_sample = 66
            num_to_sample_test = 6
        elif 0.04 <= ratio < 0.1 and len(all_frames) > 110:
            num_to_sample = 100
            num_to_sample_test = 10
        elif 0.1 <= ratio < 0.3 and len(all_frames) > 220:
            num_to_sample = 200
            num_to_sample_test = 20
        elif 0.3 <= ratio and len(all_frames) > 330:
            num_to_sample = 300
            num_to_sample_test = 30
        else:
            num_to_sample = int(len(all_frames) * 0.7)
            num_to_sample_test = int(len(all_frames) * 0.1)
        print(f" 🚀 [INFO]  Number of frames to sample: {num_to_sample}, test frames: {num_to_sample_test}")
        number_of_gas = args.number_of_gas
        if num_to_sample and num_to_sample_test:
            frames_train, idx = selector(all_frames, number=num_to_sample, sort=sort, species=species, seed=42)
            s_outf = os.path.join(parent_dir, f'{cifid}_sample{num_to_sample}.xyz')
            write(s_outf, frames_train)
            print(f" 🚀 [INFO]  Train sample: {num_to_sample} frames from {len(all_frames)} total frames")
            print(f"Selected frames: saved to {s_outf}")
            print(f"Sorted frames saved to {s_outf}")

            idx_left = set(np.arange(len(all_frames))) - set(idx)       
            left_frames = []
            for idx, atom in enumerate(all_frames):
                if idx in idx_left:
                    left_frames.append(atom)  
            print(f" 🚀 [INFO]  Test sample: {num_to_sample_test} test frames from left {len(idx_left)} frames")
            if num_to_sample_test >= 2:
                frames_test, _ = selector(left_frames, number=num_to_sample_test, sort=sort, species=species, seed=42)
            else:
                frames_test, _ = selector(left_frames, number=num_to_sample_test, sort=False, species=species, seed=42)
            t_outf = os.path.join(parent_dir, f'{cifid}_sampletest.xyz')
            write(t_outf, frames_test)
            
            print(f"Selected test frames: saved to {t_outf}")
            if sort:
                print(f"Sorted test frames saved to {t_outf}")
    
    # Sample frames
    run_sample = args.run_sample
    if run_sample:
        devi_files = sorted(glob(f'{parent_dir}/devi_{suffix}/*.xyz'))
        devi_files = [f for f in devi_files if os.path.getsize(f) > 0]       
        all_files = devi_files
        print(f" 🚀 [INFO]  Read files from: {all_files}")
        all_frames = []
        for f in all_files:
            traj = read(f, index='::5')
            print(f" 🚀 [INFO]  select {len(traj)} frames from {f}")
            all_frames += traj
        print(f"{len(all_frames)} frames from deviation evaluation")
        sort = args.sort_sample
        num_to_sample = args.num_to_sample
        
        if 2 <= len(all_frames) <= num_to_sample:
            print(f" 🚀 [INFO]  Not enough frames to sample: {len(all_frames)} < {num_to_sample}")
            if args.sample_test:
                num_to_sample_test = args.num_to_sample_test
                if len(all_frames) <= num_to_sample_test*2:
                    print(f" 🚀 [INFO]  Not enough frames to sample test: {len(all_frames)} < {num_to_sample_test*2}")
                    num_to_sample_test = int(len(all_frames)/2)
                    num_to_sample = len(all_frames)-num_to_sample_test
                    frames_train, idx = selector(all_frames, number=num_to_sample, sort=sort, species=species, seed=42)
                    s_outf = os.path.join(parent_dir, f'{cifid}_sample{num_to_sample}.xyz')
                    write(s_outf, frames_train)
                    print(f" 🚀 [INFO]  Train sample: {num_to_sample} frames from {len(all_frames)} total frames")
                    print(f"Selected frames: saved to {s_outf}")
                    if sort:
                        print(f"Sorted frames saved to {s_outf}")
                    idx_left = set(np.arange(len(all_frames))) - set(idx)
                    left_frames = []
                    for idx, atom in enumerate(all_frames):
                        if idx in idx_left:
                            left_frames.append(atom)
                    frames_test, _ = selector(left_frames, number=num_to_sample_test, sort=sort, species=species, seed=42)
                    t_outf = os.path.join(parent_dir, f'{cifid}_sampletest.xyz')
                    write(t_outf, frames_test)
                    print(f" 🚀 [INFO]  Test sample: {num_to_sample_test} test frames from left {len(idx_left)} frames")
                    print(f"Selected test frames: saved to {t_outf}")
                    if sort:
                        print(f"Sorted test frames saved to {t_outf}")
                else:
                    num_to_sample = len(all_frames)-num_to_sample_test
                    frames_train, idx = selector(all_frames, number=num_to_sample, sort=sort, species=species, seed=42)
                    s_outf = os.path.join(parent_dir, f'{cifid}_sample{num_to_sample}.xyz')
                    write(s_outf, frames_train)
                    print(f" 🚀 [INFO]  Train sample: {num_to_sample} frames from {len(all_frames)} total frames")
                    print(f"Selected frames: saved to {s_outf}")
                    if sort:
                        print(f"Sorted frames saved to {s_outf}")
                    idx_left = set(np.arange(len(all_frames))) - set(idx)
                    left_frames = []
                    for idx, atom in enumerate(all_frames):
                        if idx in idx_left:
                            left_frames.append(atom)
                    frames_test, _ = selector(left_frames, number=num_to_sample_test, sort=sort, species=species, seed=42)
                    t_outf = os.path.join(parent_dir, f'{cifid}_sampletest.xyz')
                    write(t_outf, frames_test)
                    print(f" 🚀 [INFO]  Test sample: {num_to_sample_test} test frames from left {len(idx_left)} frames")
                    print(f"Selected test frames: saved to {t_outf}")
                    if sort:
                        print(f"Sorted test frames saved to {t_outf}")
        elif len(all_frames) < 2:
            print(f" 🚀 [INFO]  Not enough frames to sample: {len(all_frames)} < {num_to_sample}")
            print(f" 🚀 [INFO]  successfully converg")
        else:
            frames_train, idx = selector(all_frames, number=num_to_sample, sort=sort, species=species, seed=42)
            s_outf = os.path.join(parent_dir, f'{cifid}_sample{num_to_sample}.xyz')
            write(s_outf, frames_train)
            print(f" 🚀 [INFO]  Train sample: {num_to_sample} frames from {len(all_frames)} total frames")
            print(f"Selected frames: saved to {s_outf}")
            if sort:
                print(f"Sorted frames saved to {s_outf}")
            if args.sample_test:
                idx_left = set(np.arange(len(all_frames))) - set(idx)
                num_to_sample_test = args.num_to_sample_test
                
                left_frames = []
                for idx, atom in enumerate(all_frames):
                    if idx in idx_left:
                        left_frames.append(atom)

                frames_test, _ = selector(left_frames, number=num_to_sample_test, sort=sort, species=species, seed=42)
                t_outf = os.path.join(parent_dir, f'{cifid}_sampletest.xyz')
                write(t_outf, frames_test)
                print(f" 🚀 [INFO]  Test sample: {num_to_sample_test} test frames from left {len(idx_left)} frames")
                print(f"Selected test frames: saved to {t_outf}")
                if sort:
                    print(f"Sorted test frames saved to {t_outf}")
    
    run_sample_gas = args.run_sample_gas
    number_of_gas = args.number_of_gas
    if run_sample_gas:
        md_files = sorted(glob(f'{parent_dir}/md_{suffix}/*.xyz'))
        opt_files = sorted(glob(f'{parent_dir}/opt_{suffix}/*.xyz'))
        md_frames = []
        #for f in md_files:
        #    traj = read(f, index=':')[1:]
        #    initial_length = len(traj[0])
        #    md_frames += traj
        all_files = md_files + opt_files
        print(f" 🚀 [INFO]  Read files from: {all_files}")
        all_frames = []
        for f in all_files:
            traj = read(f, index='::5')[1:]
            if f in md_files:
                md_frames += traj
                ## random select 1/4 of the frames
                num_frames = len(traj)
            all_frames += traj
        print(f"{len(md_frames)} frames from MD and {len(all_frames)-len(md_frames)} frames from optimization")

        sort = args.sort_sample
        num_to_sample = args.num_to_sample
        frames_train, idx = selector_tail(all_frames, number=num_to_sample, sort=sort, species=species, seed=42, n_tail=number_of_gas)
        s_outf = os.path.join(parent_dir, f'{cifid}_sample{num_to_sample}.xyz')
        write(s_outf, frames_train)
        print(f" 🚀 [INFO]  Train sample: {num_to_sample} frames from {len(all_frames)} total frames")
        print(f"Selected frames: saved to {s_outf}")
        if sort:
            print(f"Sorted frames saved to {s_outf}")
        if args.sample_test:
            idx_left = set(np.arange(len(all_frames))) - set(idx)
            num_to_sample_test = args.num_to_sample_test
            
            left_frames = []
            for idx, atom in enumerate(all_frames):
                if idx in idx_left:
                    left_frames.append(atom)

            frames_test, _ = selector_tail(left_frames, number=num_to_sample_test, sort=sort, species=species, seed=42, n_tail=number_of_gas)
            t_outf = os.path.join(parent_dir, f'{cifid}_sampletest.xyz')
            write(t_outf, frames_test)
            print(f" 🚀 [INFO]  Test sample: {num_to_sample_test} test frames from left {len(idx_left)} frames")
            print(f"Selected test frames: saved to {t_outf}")
            if sort:
                print(f"Sorted test frames saved to {t_outf}")

                    
                    
    
