from typing import Union, List, Tuple

from ase import Atoms
import numpy as np

import MDAnalysis as mda

from molecule import add_molecule


def get_accessible_positions(
    structure: Atoms,
    grid_spacing: float = 0.10,
    cutoff_distance: float = 1.2,
    min_interplanar_distance: float = 6.0,
) -> Tuple[np.ndarray, Atoms]:
    """Calculate the accessible positions in the structure

    Parameters
    ----------
    structure : Atoms
        The structure to calculate the accessible positions.
    grid_spacing : float, default=0.10
        Spacing of the grid for possible gas insertion points, in angstroms.
    cutoff_distance : float, default=1.20
        When the distance between framework atoms and the gas molecule is less than this value, the insertion is rejected. In angstroms.
    min_interplanar_distance : float, default=6.0
        When the interplanar distance of the framework is less than this value, a supercell is constructed. In angstroms.

    Returns
    -------
    Tuple[np.ndarray, Atoms]
        The result dictionary containing the following
        - pos_grid : np.ndarray
            The grid positions.
        - energy_grid : np.ndarray
            The energy grid.
        - idx_accessible_pos : np.ndarray
            The indices of the accessible positions.
        - accessible_pos : np.ndarray
            The accessible positions.
        - structure : ase.Atoms
            The supercell structure.
    """
    # get the supercell structure
    cell_volume = structure.get_volume()
    cell_vectors = np.array(structure.cell)
    dist_a = cell_volume / np.linalg.norm(np.cross(cell_vectors[1], cell_vectors[2]))
    dist_b = cell_volume / np.linalg.norm(np.cross(cell_vectors[2], cell_vectors[0]))
    dist_c = cell_volume / np.linalg.norm(np.cross(cell_vectors[0], cell_vectors[1]))
    plane_distances = np.array([dist_a, dist_b, dist_c])
    supercell = np.ceil(min_interplanar_distance / plane_distances).astype(int)
    if np.any(supercell > 1):
        print(
            f"Making supercell: {supercell} to prevent interplanar distance < {min_interplanar_distance}"
        )
    structure = structure.repeat(supercell)
    # get position for grid
    grid_size = np.ceil(np.array(structure.cell.cellpar()[:3]) / grid_spacing).astype(
        int
    )
    indices = np.indices(grid_size).reshape(3, -1).T  # (G, 3)
    pos_grid = indices.dot(cell_vectors / grid_size)  # (G, 3)
    # get positions for atoms
    pos_atoms = structure.get_positions()  # (N, 3)
    # distance matrix
    dist_matrix = mda.lib.distances.distance_array(
        pos_grid, pos_atoms, box=structure.cell.cellpar()
    )  # (G, N)

    # calculate the accessible positions
    min_dist = np.min(dist_matrix, axis=1)  # (G,)
    idx_accessible_pos = np.where(min_dist > cutoff_distance)[0]

    # result
    ret = {
        "pos_grid": pos_grid,
        "idx_accessible_pos": idx_accessible_pos,
        "accessible_pos": pos_grid[idx_accessible_pos],
        "structure": structure,
    }
    return ret


def add_gas_in_accessible_positions(
    structure: Atoms,
    gas: Union[Atoms, List[Atoms]],
    grid_spacing: float = 0.1,
    cutoff_distance: float = 1.3,
    min_interplanar_distance: float = 6.0,
) -> np.ndarray:
    """Add gas molecules in the accessible positions

    Parameters
    ----------
    structure : Atoms
        The structure to insert the gas molecules.
    gas : Union[Atoms, List[Atoms]]
        The gas molecule to insert, or a list of gas molecules.
    grid_spacing : float, default=0.15
        Spacing of the grid for possible gas insertion points, in angstroms.
    cutoff_distance : float, default=1.50
        When the distance between framework atoms and the gas molecule is less than this value, the insertion is rejected. In angstroms.
    min_interplanar_distance : float, default=6.0
        When the interplanar distance of the framework is less than this value, a supercell is constructed. In angstroms.

    Returns
    -------
    Atoms
        The structure with the gas molecules inserted
    """
    if isinstance(gas, Atoms):
        gas_list = [gas]
    elif isinstance(gas, list) or isinstance(gas, tuple):
        gas_list = gas
    else:
        raise ValueError(
            "Gas should be an ASE Atoms object or a list of ASE Atoms objects"
        )

    for gas_molecule in gas_list:
        # calculate the accessible positions
        ret = get_accessible_positions(
            structure=structure,
            grid_spacing=grid_spacing,
            cutoff_distance=cutoff_distance,
            min_interplanar_distance=min_interplanar_distance,
        )
        accessible_pos = ret["accessible_pos"]
        structure = ret["structure"]
        if len(accessible_pos) < 1:
            raise ValueError(
                "Number of accessible positions is less than the number of gas"
            )
        random_idx = np.random.choice(len(accessible_pos))
        gas_molecule = add_molecule(
            gas_molecule, rotate=True, translate=accessible_pos[random_idx]
        )
        structure += gas_molecule
        structure.wrap()
    return structure
