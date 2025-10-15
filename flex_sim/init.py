import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import molecule
from .grid import get_accessible_positions
from .molecule import add_molecule
from .u_md import GeometryOptimization
from mace.calculators import MACECalculator

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


def insert_and_optimize_gas(
    ciffile: str,
    gas_name: str,
    modelpath: str,
    num_of_gas: int,
    grid_spacing: float = 0.2,
    cutoff_distance: float = 2.0,
    min_interplanar_distance: float = 10.0,
    random_seed: int = 42,
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
        gas_molecule = add_molecule(gas, rotate=True, translate=pos)
        structure += gas_molecule
        structure.wrap()

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