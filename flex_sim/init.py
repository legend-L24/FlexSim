import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import molecule
from .grid import get_accessible_positions
from .molecule import add_molecule
from .u_md import GeometryOptimization
from mace.calculators import MACECalculator
from ase.geometry import get_distances


def farthest_point_sampling_pbc_vectorized(points: np.ndarray, atoms: Atoms, n_samples: int,
                                           random_seed: int = 0, min_distance: float = 2.0) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) under PBC using vectorized numpy operations,
    ensuring sampled points are at least `min_distance` away from any atom in `atoms`.

    Parameters
    ----------
    points : np.ndarray, shape (M,3)
        Candidate points
    atoms : ASE Atoms object
        Provides cell vectors and PBC
    n_samples : int
        Number of points to sample
    random_seed : int
        Random seed
    min_distance : float
        Minimum allowed distance between sampled points and existing atoms (Å)

    Returns
    -------
    np.ndarray, shape (n_samples,3)
        Sampled points
    """
    np.random.seed(random_seed)
    M = points.shape[0]
    if n_samples > M:
        raise ValueError("n_samples must be <= number of points")

    cell = atoms.get_cell().array  # (3,3)
    pbc = atoms.get_pbc()
    atom_positions = atoms.get_positions()

    # 计算 points 到原子的最短 PBC 距离
    diff = points[:, None, :] - atom_positions[None, :, :]  # (M, N_atoms, 3)
    for i in range(3):
        if pbc[i]:
            diff[..., i] -= np.round(diff[..., i] / cell[i, i]) * cell[i, i]
    distances = np.linalg.norm(diff, axis=-1)
    min_to_atoms = np.min(distances, axis=1)

    # 过滤掉离原子太近的点
    valid_mask = min_to_atoms >= min_distance
    valid_points = points[valid_mask]
    if len(valid_points) < n_samples:
        raise ValueError(f"Not enough valid points after filtering (need {n_samples}, found {len(valid_points)}).")

    points = valid_points
    M = len(points)

    # 初始化最远点采样
    selected_idx = [np.random.randint(M)]
    delta = points - points[selected_idx[0]]
    for i in range(3):
        if pbc[i]:
            delta[:, i] -= np.round(delta[:, i] / cell[i, i]) * cell[i, i]
    min_distances = np.linalg.norm(delta, axis=1)

    for _ in range(1, n_samples):
        idx = np.argmax(min_distances)
        selected_idx.append(idx)

        delta = points - points[idx]
        for i in range(3):
            if pbc[i]:
                delta[:, i] -= np.round(delta[:, i] / cell[i, i]) * cell[i, i]
        dist = np.linalg.norm(delta, axis=1)
        min_distances = np.minimum(min_distances, dist)

    return points[selected_idx]


'''
def farthest_point_sampling_pbc_vectorized(points: np.ndarray, atoms: Atoms, n_samples: int, random_seed: int = 0) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) under PBC using vectorized numpy operations.

    Parameters
    ----------
    points : np.ndarray, shape (M,3)
        Candidate points
    atoms : ASE Atoms object
        Provides cell vectors and PBC
    n_samples : int
        Number of points to sample
    random_seed : int
        Random seed

    Returns
    -------
    np.ndarray, shape (n_samples,3)
        Sampled points
    """
    np.random.seed(random_seed)
    M = points.shape[0]
    if n_samples > M:
        raise ValueError("n_samples must be <= number of points")

    cell = atoms.get_cell().array  # (3,3)
    pbc = atoms.get_pbc()

    selected_idx = [np.random.randint(M)]

    # 初始化最小距离
    delta = points - points[selected_idx[0]]
    # Apply minimum image convention
    for i in range(3):
        if pbc[i]:
            delta[:, i] -= np.round(delta[:, i] / cell[i, i]) * cell[i, i]
    min_distances = np.linalg.norm(delta, axis=1)

    for _ in range(1, n_samples):
        idx = np.argmax(min_distances)
        selected_idx.append(idx)

        # 更新最小距离，矢量化
        delta = points - points[idx]
        for i in range(3):
            if pbc[i]:
                delta[:, i] -= np.round(delta[:, i] / cell[i, i]) * cell[i, i]
        dist = np.linalg.norm(delta, axis=1)
        min_distances = np.minimum(min_distances, dist)

    return points[selected_idx]
'''

def insert_and_optimize_gas(
    ciffile: str,
    gas_name: str,
    modelpath: str,
    num_of_gas: int,
    grid_spacing: float = 0.2,
    cutoff_distance: float = 2.0,
    min_interplanar_distance: float = 10.0,
    random_seed: int = 44,#42,
    if_optimized: bool = True,
    Fmax: float = 0.01,
    stepMax: int = 200,
    device: str = 'cuda',
    logfile: str = 'opt.log'
) -> Atoms:
    """
    Insert gas molecules into a porous crystal structure and optionally optimize the geometry.

    Parameters
    ----------
    ciffile : str
        Path to the CIF file of the host structure.
    gas_name : str
        Name of the gas molecule (e.g., "H2O", "CO2") recognized by ASE's molecule().
    modelpath : str
        Path to the MACE model file for optimization.
    num_of_gas : int
        Number of gas molecules to insert.
    grid_spacing : float, optional
        Grid spacing for accessible volume calculation (Å). Default: 0.2.
    cutoff_distance : float, optional
        Minimum distance from framework atoms to consider a point accessible (Å). Default: 2.0.
    min_interplanar_distance : float, optional
        Minimum interplanar distance for grid generation (Å). Default: 10.0.
    random_seed : int, optional
        Random seed for reproducible sampling. Default: 42.
    if_optimized : bool, optional
        Whether to perform geometry optimization after insertion. Default: True.
    Fmax : float, optional
        Force convergence criterion for optimization (eV/Å). Default: 0.01.
    stepMax : int, optional
        Maximum number of optimization steps. Default: 200.
    device : str, optional
        Device for MACE calculator ('cpu' or 'cuda'). Default: 'cuda'.
    logfile : str, optional
        Log file name for optimization. Default: 'opt.log'.

    Returns
    -------
    ASE Atoms object
        Final structure (optimized if requested, otherwise initial with inserted gas).
    """
    # Load host structure and gas molecule
    structure = read(ciffile)
    gas = molecule(gas_name)

    # Fix all original framework atoms during optimization
    fix_indices = list(range(len(structure)))

    # Find accessible positions in the pores
    ret = get_accessible_positions(
        structure=structure,
        grid_spacing=grid_spacing,
        cutoff_distance=cutoff_distance,
        min_interplanar_distance=min_interplanar_distance,
    )

    # Select well-separated insertion sites using farthest point sampling
    selected_positions = farthest_point_sampling_pbc_vectorized(
        points=ret['accessible_pos'],
        atoms=structure,
        n_samples=num_of_gas,
        random_seed=random_seed
    )
    # Insert gas molecules at selected positions
    for pos in selected_positions:
        attempt = 0
        max_attempts = 20
        inserted = False
        threshold_insert = 1.5
        #print(f"{attempt+1}. 尝试插入 {gas_name} 于位置 ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        while attempt < max_attempts and not inserted:
            candidate_mol = add_molecule(gas, rotate=True, translate=pos)
            original_positions = structure.get_positions()
            new_mol_positions = candidate_mol.get_positions()
            #print(original_positions.shape, new_mol_positions.shape)

            # 计算距离
            _, distances = get_distances(
                original_positions,
                new_mol_positions,
                cell=structure.get_cell(),
                pbc=structure.get_pbc()
            )

            if len(distances) == 0:
                print("⚠️ Warning: distances 数组为空，跳过该点。")
                attempt += 1
                continue

            minimum_distance = np.min(distances)
            print(minimum_distance)
            if minimum_distance > threshold_insert:
                structure += candidate_mol
                structure.wrap()
                inserted = True

            attempt += 1

        if not inserted:
            print(f"❌ 插入 {gas_name} 失败于位置 ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")


    initial_structure = structure.copy()

    # Optionally optimize the structure
    if if_optimized:
        mace_calc = MACECalculator(model_paths=[modelpath], device=device)
        final_structure, _ = GeometryOptimization(
            atoms=structure,
            Fmax=Fmax,
            stepMax=stepMax,
            calc=mace_calc,
            save_traj=False,
            outputatoms=True,
            logfile=logfile,
            fix_indices=fix_indices
        )
        return final_structure
    else:
        return initial_structure