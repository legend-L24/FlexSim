from flex_sim.u_md import simpleMD, GeometryOptimization
from flex_sim.u_selector import selector, selector_tail
from ase.io import read, write
import random
from collections import Counter
import os
import numpy as np
from glob import glob
import argparse
from mace.calculators import MACECalculator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MD simulation and optimization")
    parser.add_argument('--cif', type=str, required=True, help='Input CIF file path')
    parser.add_argument('--build_new_ligand', action='store_true', help='Build a new ligand from the CIF file')
    parser.add_argument('--scaling', type=float, default=1.0, help='The scaling factor for vesta cutoff')
    parser.add_argument('--cell_type', type=str, default="supercell", help='The type of cell to use (cubic or supercell)')
    parser.add_argument('--cutoff', type=float, default=None, help='Cutoff distance for supercell construction (Å)')
    parser.add_argument('--cubic_method', type=str, default="auto", help='Fixed cubic length or compute based on the radius of molecule')
    parser.add_argument('--threshold', type=float, default=4, help="Radius to add if cubic_method is auto")
    parser.add_argument('--cell_length', type=float, default=None, help='Target cubic cell length (Å)')
    parser.add_argument('--suffix', type=str, default=None, help='Folder suffix for md (write) and sample (read)')
    parser.add_argument('--run_md', action='store_true', help='Run MD simulation')
    parser.add_argument('--md_steps', type=int, default=10000, help='Number of MD steps')
    parser.add_argument('--num_to_geo_opt', type=int, default=100, help='Random seed for frame selection')
    parser.add_argument('--run_sample', action='store_true', help='Do sampling or not')
    parser.add_argument('--sort_sample', action='store_true', help='Sort sampled frames or not')
    parser.add_argument('--num_to_sample', type=int, default=300, help='Number of frames to sample')
    parser.add_argument('--sample_test', action='store_true', help='Do test sampling or not')
    parser.add_argument('--num_to_sample_test', type=int, default=200, help='Number of test frames to sample')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for MACE calculation')
    parser.add_argument('--restart_geoopt', action='store_true', help='Run model deviation evaluation')
    parser.add_argument('--run_sample_gas', action='store_true', help='Do sampling or not')
    parser.add_argument('--number_of_gas', type=int, default=3, help='Do sampling or not')
    parser.add_argument('--cueq', action='store_true', help='Use cueq acceleration for MD simulation')
    parser.add_argument('--modelpath', type=str, default='/users/lyutao/project/mil120_221/mace-medium-density-agnesi-stress.model', help='Path to the MACE model')
    
    args = parser.parse_args()
    
    device = args.device
     
    model_path = args.modelpath
    cueq_choice = args.cueq
    
    
    
    incif_fullpath = args.cif
    parent_dir = os.path.dirname(incif_fullpath)
    cifid = os.path.basename(incif_fullpath).split('.')[0]
    suffix = args.suffix
    species = set(read(incif_fullpath).get_chemical_symbols())
    
    outxyz = incif_fullpath
    print(f" 🚀 [INFO] Using original CIF file: {outxyz}")
    
    restart_geoopt = args.restart_geoopt
    if restart_geoopt:
        mace_calc = MACECalculator(model_paths=['/users/lyutao/project/mil120_221/mace-medium-density-agnesi-stress.model'], device=device)  
        md_files = []
        num_geo_opt = args.num_to_geo_opt
        for k in [100, 300, 500, 600]:
            print(f" 🚀 [INFO] Running MD simulation at {k}K")
            init_conf = read(outxyz, index=':-1')
            outf = f"{parent_dir}/md_{suffix}/{cifid}_{k}K.xyz"
            md_files.append(outf)
        md_frames = []
        for f in md_files:
            traj = read(f, index=':-1')[1:]
            md_frames += traj
        frames_to_go, idx_go = selector(md_frames, number=num_geo_opt, sort=False, species=species, seed=6666)
        print(f" 🚀 [INFO]  {num_geo_opt} frames selected for geometry optimization. ")
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
        
    # Run MD simulation
    run_md = args.run_md
    if run_md:
        if cueq_choice:
            mace_calc = MACECalculator(model_paths=[model_path], device="cuda", cu_eq=True)  
        else:
            mace_calc = MACECalculator(model_paths=[model_path], device="cuda")  
        
        num_geo_opt = args.num_to_geo_opt
        
        md_files = []
        os.makedirs(os.path.join(parent_dir, f'md_{suffix}'), exist_ok=True)
        for k in [100, 300, 500, 600]:
            print(f" 🚀 [INFO] Running MD simulation at {k}K")
            init_conf = read(outxyz, index=':-1')
            
            outf = f"{parent_dir}/md_{suffix}/{cifid}_{k}K.xyz"
            md_files.append(outf)
            fig = simpleMD(init_conf, temp=k, calc=mace_calc, fname=outf, s=10, T=args.md_steps)
            fig.savefig(f"{outf.split('.')[0]}.png")

        md_frames = []
        for f in md_files:
            traj = read(f, index=':-1')[1:]
            md_frames += traj
    
        frames_to_go, idx_go = selector(md_frames, number=num_geo_opt, sort=False, species=species, seed=6666)
        print(f" 🚀 [INFO]  {num_geo_opt} frames selected for geometry optimization. ")
        flgs = []
        opt_files = []
        os.makedirs(os.path.join(parent_dir, f'opt_{suffix}'), exist_ok=True)
        for i, frame in zip(idx_go, frames_to_go):
            outf = f"{parent_dir}/opt_{suffix}/opt_{i}.xyz"
            logf = f"{parent_dir}/opt_{suffix}/opt.log"
            flg = GeometryOptimization(frame, outf, logf, 5, 0.01, 500, calc=mace_calc)
            flgs.append(flg)
            opt_files.append(outf)
        print(f"Geometry optimization finished! {Counter(flgs)}")

    # Sample frames
    run_sample = args.run_sample
    if run_sample:
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
            traj = read(f, index=':-1')[1::2]
            if f in md_files:
                md_frames += traj
                ## random select 1/4 of the frames
                num_frames = len(traj)
                print(f" 🚀 [INFO]  select {num_frames} frames from {f}")
            all_frames += traj
        print(f"{len(md_frames)} frames from MD and {len(all_frames)-len(md_frames)} frames from optimization")

        sort = args.sort_sample
        num_to_sample = args.num_to_sample
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
        # Sample gas phase
            # Sample frames
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
            traj = read(f, index=':-1')[1::2]
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

