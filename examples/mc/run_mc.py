from ase import Atoms
from ase.io import write, read
import numpy as np
import random
import math
import os
import sys
from mace.cli.eval_configs import main

def perturb_water(molecule):
    disp = np.random.normal(scale=0.1, size=3)
    molecule.translate(disp)
    axis = np.random.rand(3) - 0.5
    angle = np.random.normal(scale=np.pi / 10)
    molecule.rotate(angle, v=axis, center='COM')
    atoms = molecule.get_positions()
#    atoms[0] += np.random.normal(scale=0.03, size=3)
    atoms[1] += np.random.normal(scale=0.03, size=3)
    atoms[2] += np.random.normal(scale=0.03, size=3)
    molecule.set_positions(atoms)
    return molecule

def run_eval(temp_xyz, model_path, temp_eval, index):
    sys.argv = [
        "eval_configs.py",
        f"--configs={temp_xyz}",
        f"--model={model_path}",
        f"--output={temp_eval}"
    ]
    try:
        main()
    except Exception as e:
        print(f"❌ Error in block {index}: {e}")
        return None

    if os.path.exists(temp_eval):
        with open(temp_eval, "r") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                for item in lines[1].strip().split():
                    if item.startswith("MACE_energy="):
                        try:
                            energy = float(item.split("=")[-1])
                            return energy
                        except ValueError:
                            print("❌ MACE_energy format invalid.")
    print(f"❌ Failed to read energy from {temp_eval}")
    return None

def main_mc():
    mol = read("h2o.xyz")
    mol.set_cell([31.28, 31.28, 31.28])
    mol.set_pbc([True, True, True])
    mol.wrap()

    T = 300
    beta = 1 / (8.3145e-3 * T)
    nsteps = 10000
    energy = None
    model_path = "./MACE.model"

    accepted = 0
    accepted_energies = []

    with open("energy.dat", "w") as energy_file, open("mc.xyz", "w") as mc_xyz_file:
        for step in range(nsteps):
            new_mol = mol.copy()
            new_mol = perturb_water(new_mol)
            new_mol.set_cell([31.28, 31.28, 31.28])
            new_mol.set_pbc([True, True, True])
            new_mol.wrap()

            temp_xyz = f"water_{step}.xyz"
            temp_eval = f"energy_{step}.txt"

            write(temp_xyz, new_mol)
            new_energy = run_eval(temp_xyz, model_path, temp_eval, step)

            if os.path.exists(temp_xyz):
                os.remove(temp_xyz)
            if os.path.exists(temp_eval):
                os.remove(temp_eval)

            if new_energy is None:
                print(f"Step {step}: failed to get energy, skipping.")
                continue

            accept = False
            if energy is None or random.uniform(0, 1) < math.exp(-beta * 96.48*(new_energy - energy)):
                mol = new_mol
                energy = new_energy
                accept = True
                accepted += 1
                accepted_energies.append(energy)
                print(f"✅ Accepted step {step} with energy {energy:.4f}")
            else:
                print(f"❌ Rejected step {step} with energy {new_energy:.4f}")

            if accept:
                energy_file.write(f"{energy:.8f}\n")
                energy_file.flush()
                write(mc_xyz_file, mol, format="extxyz", append=True)
                mc_xyz_file.flush()

    acceptance_ratio = accepted / nsteps
    average_energy = np.mean(accepted_energies) if accepted_energies else 0.0
    energy_std = np.std(accepted_energies) if accepted_energies else 0.0

    with open("log.txt", "w") as log:
        log.write(f"Total steps: {nsteps}\n")
        log.write(f"Accepted steps: {accepted}\n")
        log.write(f"Acceptance ratio: {acceptance_ratio:.4f}\n")
        log.write(f"Average energy: {average_energy:.8f} eV\n")
        log.write(f"Energy fluctuation (std): {energy_std:.8f} eV\n")

    print(f"\n📊 MC Summary:")
    print(f"Total steps: {nsteps}")
    print(f"Accepted steps: {accepted}")
    print(f"Acceptance ratio: {acceptance_ratio:.4f}")
    print(f"Average energy: {average_energy:.8f} eV")
    print(f"Energy fluctuation (std): {energy_std:.8f} eV")

if __name__ == "__main__":
    main_mc()
