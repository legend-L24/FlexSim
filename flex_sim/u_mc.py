from time import time
import numpy as np
from ase import Atoms, units
from ase.io import read, write
from mace.calculators import MACECalculator
from ase.geometry import get_distances
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.logger import MDLogger
from time import time

import numpy as np
from ase import Atoms
from ase import units
from ase.optimize.optimize import Dynamics
from ase.io.trajectory import Trajectory
from pathlib import Path

from flex_sim.grid import get_accessible_positions
from flex_sim.molecule import add_molecule

# This is the code to generate the water loading configurations by widom insertion MC method.

def bestpos_insertion(ciffile, gasfile, calc, n_insertions=1000,min_interplanar_distance=6,  grid_spacing=0.2, cutoff_distance=1.5):
    '''
    
    Ciffile: Path object from the pathlib module, pointing to the host structure cif file.
    calc: ASE calculator object, e.g. MACECalculator
    
    '''
    # check if the ciffiles is Path object
    if not isinstance(ciffile, Path):
        ciffile = Path(ciffile)
    
    
    structure = read(ciffile)
    gas = read(gasfile)
    
    loadingcif = ciffile.parent / ciffile.name.replace(".cif", "_loading.cif")
    
    ret = get_accessible_positions(
        structure=structure,
        grid_spacing=grid_spacing,
        cutoff_distance=cutoff_distance,
        min_interplanar_distance=min_interplanar_distance,
    )

    pos_grid = ret["pos_grid"]
    idx_accessible_pos = ret["idx_accessible_pos"]
    structure = ret["structure"] 
    print(
        f"Number of accessible positions: {len(idx_accessible_pos)} out of total {len(pos_grid)}"
            )
    random_indices = np.random.choice(len(pos_grid), size=n_insertions, replace=True)

    optimal_energy = 0
    optimal_config = None
    
    structure.calc = calc
    mof_energy = structure.get_potential_energy()
    gas.calc = calc
    gas_energy = gas.get_potential_energy()
    
    
    for i, rand_idx in enumerate((random_indices)):
        if rand_idx not in idx_accessible_pos:
            continue
        pos = pos_grid[rand_idx]
        added_gas = add_molecule(gas, rotate=True, translate=pos)
        structure_with_gas = structure + added_gas
        structure_with_gas.wrap() 
        structure_with_gas.calc = calc
        total_energy = structure_with_gas.get_potential_energy() #[eV]
        if not optimal_energy:
            optimal_energy = total_energy
            optimal_config = structure_with_gas
        elif optimal_energy > total_energy:
            optimal_energy = total_energy
            optimal_config = structure_with_gas 
    interaction_energy = optimal_energy - mof_energy - gas_energy   
    if interaction_energy < -1.2:
        print('Warning: Unusually strong interaction energy detected:', interaction_energy)
    write(loadingcif, optimal_config)
    print(f"Optimal loading configuration saved to {loadingcif} with interaction energy {interaction_energy:.6f} eV")


# This is the code for MC/MD hybrid with simple random reinsertion of water molecules.


# ========= 辅助函数 =========

# ========================
# 0. Wrap Water
# ========================

def wrap_water_to_oxygen(atoms, o_local_idx):
    """
    将水分子的三个原子通过最小镜像约定，以氧原子为中心对齐。
    输出顺序与 atoms[-3:] 完全一致。
    """
    pos = atoms.positions[-3:].copy()   # [A, B, C] —— 顺序与输入一致
    o_pos = pos[o_local_idx]
    wrapped = pos.copy()                # 初始化为原始顺序

    for i in range(3):
        if i == o_local_idx:
            continue
        # 计算从 O 到第 i 个原子的 MIC 向量
        vec_mic, dist = get_distances([o_pos], [pos[i]], cell=atoms.cell, pbc=atoms.pbc)
        if dist[0,0] > 1.2 or dist[0,0] < 0.5:
            print(f"Warning: Weired distance {dist[0,0]:.2f} Å between O and atom index {i} in water molecule.")
        wrapped[i] = o_pos + vec_mic[0, 0]  # 只移动非氧原子，保持数组顺序

    return wrapped  # 顺序仍为 [A, B, C]，其中一个是 O，两个是 H
# ========================
# 4. Random rotation
# ========================

def random_rotation_matrix(rng):
    """随机生成3x3旋转矩阵（均匀分布于SO(3)）"""
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
# ========================
# 3. Build water from snapshot geometry
# ========================
def build_water_positions(o_pos, rotation_matrix, original_water_positions, o_local_index):
    orig = np.array(original_water_positions)
    current_o = orig[o_local_index]
    local_coords = orig - current_o
    rotated_local = local_coords @ rotation_matrix.T
    return o_pos + rotated_local

def identify_water_o_index(symbols):
    """识别水分子中O的索引"""
    symbols = list(symbols)
    if symbols.count('O') == 1 and symbols.count('H') == 2:
        return symbols.index('O')
    else:
        raise ValueError(f"Last 3 atoms are not H2O: {symbols}")

def generate_random_o_position(rng, cell):
    """在晶胞内随机生成O原子位置（支持非正交晶格）"""
    frac = rng.random(3)
    return np.dot(frac, cell)

# ========= 主 reinsertion 函数 =========
def simple_reinsertion_once(atoms, calc, temp, rng, min_dist=2.0, n_trials=200):
    """
    尝试最多 n_trials 次随机生成水分子位置，
    并用 ASE 的 get_distances() 检查周期性最近距离。
    """
    atoms.calc = calc
    U_old = calc.get_potential_energy(atoms)
    n = len(atoms)
    if n < 3:
        return False

    # 假设最后3个原子是水
    original_water = atoms[-3:].copy()
    original_symbols = original_water.get_chemical_symbols()
    o_idx = identify_water_o_index(original_symbols)
    original_positions = wrap_water_to_oxygen(original_water, o_idx)
    host = atoms[:-3].copy()
    frame_positions = host.get_positions()
    beta = 1.0 / (units.kB * temp)
    count_trial = 0
    while count_trial < n_trials:
        # 随机生成新水位置
        #t0 = time()
        o_trial = generate_random_o_position(rng, host.cell)
        R = random_rotation_matrix(rng)
        new_positions = build_water_positions(o_trial, R, original_positions, o_idx)
        #t1 = time()
        # ====== ✅ 一步计算最小周期性距离 ======
        _, dist = get_distances(new_positions, frame_positions, cell=atoms.cell, pbc=atoms.pbc)
        min_distance = np.min(dist)

        if min_distance < min_dist:
            continue  # 太近，跳过
        #t2 = time()
        # ====== 计算能量并判断接受 ======
        new_sys = host.copy()
        new_sys.extend(Atoms(symbols=original_symbols, positions=new_positions))
        new_sys.calc = calc
        #t3 = time()
        U_new = calc.get_potential_energy(new_sys)
        dU = U_new - U_old
        acc_prob = np.exp(-beta * dU)
        count_trial += 1
        #t4 = time()
        #print(f"Trial {count_trial}: contruction atoms time = {t1-t0}s, distance check time = {t2 - t1}s, "
        #      f"energy calc time = {t4 - t3}s, construction atoms time = {(t3 - t2)/2}s")
        if rng.random() < acc_prob:
            print(f"✅ Accepted at trial {count_trial}, ΔU = {dU:.6f} eV, "
                  f"P_acc = {acc_prob:.3e}, min_dist = {min_distance:.2f} Å")
            return True, new_positions

        if count_trial % 100 == 0:
            print(f"  ... {count_trial} trials attempted (no acceptance yet)")

    print("❌ All trials rejected.")
    return False, None

# ========= 主 reinsertion 函数 =========
def simple_reinsertion_full(atoms, calc, temp, rng, min_dist=2.0, n_trials=200):
    """
    尝试最多 n_trials 次随机生成水分子位置，
    并用 ASE 的 get_distances() 检查周期性最近距离。
    """
    atoms.calc = calc
    U_old = calc.get_potential_energy(atoms)
    n = len(atoms)
    if n < 3:
        return False

    # 假设最后3个原子是水
    original_water = atoms[-3:].copy()
    original_symbols = original_water.get_chemical_symbols()
    o_idx = identify_water_o_index(original_symbols)
    original_positions = wrap_water_to_oxygen(original_water, o_idx)
    host = atoms[:-3].copy()
    frame_positions = host.get_positions()
    beta = 1.0 / (units.kB * temp)
    if_update = False
    count_trial = 0
    while count_trial < n_trials:
        # 随机生成新水位置
        #t0 = time()
        o_trial = generate_random_o_position(rng, host.cell)
        R = random_rotation_matrix(rng)
        new_positions = build_water_positions(o_trial, R, original_positions, o_idx)
        #t1 = time()
        # ====== ✅ 一步计算最小周期性距离 ======
        _, dist = get_distances(new_positions, frame_positions, cell=atoms.cell, pbc=atoms.pbc)
        min_distance = np.min(dist)

        if min_distance < min_dist:
            continue  # 太近，跳过
        #t2 = time()
        # ====== 计算能量并判断接受 ======
        new_sys = host.copy()
        new_sys.extend(Atoms(symbols=original_symbols, positions=new_positions))
        new_sys.calc = calc
        #t3 = time()
        U_new = calc.get_potential_energy(new_sys)
        dU = U_new - U_old
        acc_prob = np.exp(-beta * dU)
        count_trial += 1
        #t4 = time()
        #print(f"Trial {count_trial}: contruction atoms time = {t1-t0}s, distance check time = {t2 - t1}s, "
        #      f"energy calc time = {t4 - t3}s, construction atoms time = {(t3 - t2)/2}s")
        if rng.random() < acc_prob:
            print(f"✅ Accepted at trial {count_trial}, ΔU = {dU:.6f} eV, "
                  f"P_acc = {acc_prob:.3e}, min_dist = {min_distance:.2f} Å")
            if_update = True
            if if_update:
                U_old = U_new
        if count_trial % 100 == 0:
            print(f"  ... {count_trial} trials attempted (no acceptance yet)")
    if if_update:
        return True, new_positions
    else:
        print("❌ All trials rejected.")
        return False, None


# ========================
# 10. MD Driver (precomputes baselines once)
# ========================
def MD_with_reinsertion(
    init_conf, temp, calc, fname, s, T,
    mc_interval=1000, seed=None,
    min_dist=1.4, n_trials=300, fraction=0.01
):
    # Precompute baseline energies ONCE
    n = len(init_conf)
    if n < 3:
        raise ValueError("System too small for H2O reinsertion.")
    

    rng = np.random.default_rng(seed)
    atoms = init_conf.copy()
    atoms.calc = calc

    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=temp, friction=fraction)
    dyn.attach(lambda: atoms.write(fname, append=True), interval=s * 10)

    fname_log = fname.replace('.xyz', '.log')
    with open(fname_log, 'w') as logfile:
        logger = MDLogger(dyn, atoms, logfile, header=True, stress=False, peratom=False)
        dyn.attach(logger, interval=s)

         # === Reinsertion acceptance statistics ===
        step = [0]
        accepted_count = [0]
        total_trials = [0]

        def mc_hook():
            step[0] += 1
            if step[0] % mc_interval == 0 and step[0] >= 2000:
                total_trials[0] += 1
                accepted, new_pos = simple_reinsertion_once(
                    atoms, calc, temp, rng,
                    min_dist=min_dist, n_trials=n_trials
                )
                if accepted:
                    accepted_count[0] += 1
                    atoms.positions[-3:] = new_pos

                    # ✅ 仅新增 2 行：对最后 3 个原子（水分子）重抽速度 + 扣除其质心运动
                    h2o = atoms[-3:].copy()  # This is a *view*, not a copy!
                    MaxwellBoltzmannDistribution(h2o, temperature_K=temp)
                    Stationary(h2o)
                    atoms.set_momenta(np.vstack([atoms.get_momenta()[:-3], h2o.get_momenta()]))
                    # ✅ 结束
                    print(f"Step {step[0]}: ✅ CBMC reinsertion ACCEPTED "
                          f"({accepted_count[0]}/{total_trials[0]} = "
                          f"{accepted_count[0]/total_trials[0]:.2%})")
                else:
                    print(f"Step {step[0]}: ❌ CBMC reinsertion REJECTED "
                          f"({accepted_count[0]}/{total_trials[0]} = "
                          f"{accepted_count[0]/total_trials[0]:.2%})")

        dyn.attach(mc_hook, interval=1)
        t0 = time()
        dyn.run(T)
        t1 = time()

    print(f"MD/CBMC finished in {(t1 - t0) / 60:.2f} minutes!")

def MD_with_reinsertion_full(
    init_conf, temp, calc, fname, s, T,
    mc_interval=100, seed=None,
    min_dist=1.4, n_trials=300
):
    # Precompute baseline energies ONCE
    n = len(init_conf)
    if n < 3:
        raise ValueError("System too small for H2O reinsertion.")
    

    rng = np.random.default_rng(seed)
    atoms = init_conf.copy()
    atoms.calc = calc

    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=temp, friction=1.0)
    dyn.attach(lambda: atoms.write(fname, append=True), interval=s * 10)

    fname_log = fname.replace('.xyz', '.log')
    with open(fname_log, 'w') as logfile:
        logger = MDLogger(dyn, atoms, logfile, header=True, stress=False, peratom=False)
        dyn.attach(logger, interval=s)

         # === Reinsertion acceptance statistics ===
        step = [0]
        accepted_count = [0]
        total_trials = [0]

        def mc_hook():
            step[0] += 1
            if step[0] % mc_interval == 0 and step[0] != 0:
                total_trials[0] += 1
                accepted, new_pos = simple_reinsertion_full(
                    atoms, calc, temp, rng,
                    min_dist=min_dist, n_trials=n_trials
                )
                if accepted:
                    accepted_count[0] += 1
                    atoms.positions[-3:] = new_pos
                    momenta = atoms.get_momenta()
                    momenta[-3:] = 0.0
                    atoms.set_momenta(momenta)
                    print(f"Step {step[0]}: ✅ CBMC reinsertion ACCEPTED "
                          f"({accepted_count[0]}/{total_trials[0]} = "
                          f"{accepted_count[0]/total_trials[0]:.2%})")
                else:
                    print(f"Step {step[0]}: ❌ CBMC reinsertion REJECTED "
                          f"({accepted_count[0]}/{total_trials[0]} = "
                          f"{accepted_count[0]/total_trials[0]:.2%})")

        dyn.attach(mc_hook, interval=1)

        t0 = time()
        dyn.run(T)
        t1 = time()

    print(f"MD/CBMC finished in {(t1 - t0) / 60:.2f} minutes!")


# ========= 测试脚本 =========

def test_MD_CBMC():
    """
    测试 MD_with_RASPA_CBMC_MACE 是否能正确运行。
    会：
      - 读取初始结构；
      - 构建 MACE calculator；
      - 跑几百步 Langevin + 每100步一次随机重插；
      - 输出 log/xyz 文件；
      - 打印接受率信息。
    """
    # ========= 用户路径 =========
    INITIAL_XYZ = "/capstor/scratch/cscs/lyutao/heat/cbmc/cau10h2o/1.xyz"
    MACE_MODEL = "/capstor/scratch/cscs/lyutao/heat/cbmc/MACE_stagetwo.model"

    # ========= 参数 =========
    temp = 300.0  # K
    total_steps = 50000  # 总 MD 步数（小规模测试）
    log_interval = 10
    mc_interval = 2000
    seed = 42

    # ========= 载入初始结构与计算器 =========
    atoms = read(INITIAL_XYZ)
    calc = MACECalculator(model_paths=MACE_MODEL, device="cuda", default_dtype="float64")

    # ========= 输出文件名 =========
    output_file = "test_reinsertion_output_long.xyz"

    print("🔹 Starting MD/CBMC hybrid test run...")
    print(f"   Structure: {len(atoms)} atoms")
    print(f"   Temperature: {temp} K")
    print(f"   Total MD steps: {total_steps}")
    print(f"   MC reinsertion every {mc_interval} steps")

    # ========= 运行主函数 =========
    MD_with_reinsertion(
        init_conf=atoms,
        temp=temp,
        calc=calc,
        fname=output_file,
        s=log_interval,
        T=total_steps,
        mc_interval=mc_interval,
        seed=seed,
        min_dist=1.5,
        n_trials=300
    )

    print("✅ Test completed successfully!")
    print(f"   Trajectory written to: {output_file}")
    print(f"   Log file: {output_file.replace('.xyz', '.log')}")

if_mcmd_test = False

## If you want to run the create loading, please copy example0.tar.gz to the flex_sim folder and unzip it first.
if_create_loading = False



if __name__ == "__main__":
    if if_mcmd_test:
        test_MD_CBMC()
        # ========= 用户定义路径 =========
        INITIAL_XYZ = "/capstor/scratch/cscs/lyutao/heat/cbmc/cau10h2o/1.xyz"
        MACE_MODEL = "/capstor/scratch/cscs/lyutao/heat/cbmc/MACE_stagetwo.model"
        
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 必须在 import torch/tf 之前！


        rng = np.random.default_rng(42)
        temp = 300.0

        print(f"Reading structure: {INITIAL_XYZ}")
        atoms = read(INITIAL_XYZ)
        atoms.calc = MACECalculator(model_paths=MACE_MODEL, device="cuda")

        n_runs = 5
        accepted = 0

        for i in range(1, n_runs + 1):
            print(f"\n=== Reinsertion attempt {i}/{n_runs} ===")
            ok = simple_reinsertion_full(
                atoms, atoms.calc, temp, rng,
                min_dist=1.4, n_trials=1000
            )
            if ok:
                accepted += 1

        print("\n========== Summary ==========")
        print(f"Total accepted: {accepted}/{n_runs} ({accepted / n_runs:.2%})")
        
    if if_create_loading:
        ciffile = "data/example0/CAU10.cif"
        gasfile = "data/example0/1h2o_best.cif"
        model_path = "data/example0/MACE_run-5555.model"
        calc = MACECalculator(model_paths=model_path, device="cuda", cueq=False)
        bestpos_insertion(
            ciffile=ciffile,
            gasfile=gasfile,
            calc=calc,
            n_insertions=300,
            min_interplanar_distance=6,
            grid_spacing=0.15,
            cutoff_distance=1.5
        )
        

