from typing import Optional, Literal
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import Trajectory
from ase.calculators.calculator import Calculator
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp

import torch
from dac_sim import DEFAULT_MODEL_PATH


class GeometryOptimization:
    def __init__(
        self,
        num_total_optimization: int = 30,
        num_internal_steps: int = 50,
        num_cell_steps: int = 50,
        fmax: float = 0.05,
        cell_relax: bool = True,
        model_path: Optional[str] = None,
        device: Literal["cuda", "cpu"] = "cuda",
        default_dtype: Literal["float32", "float64"] = "float32",
        dispersion: bool = True,
    ):
        """Geometry optimization using the FIRE algorithm

        Parameters
        ----------
        num_total_optimization : int, default=30
            The number of optimization steps including internal and cell relaxation
        num_internal_steps : int, default=50
            The number of internal steps (freezing the cell)
        num_cell_steps : int, default=50
            The number of optimization steps (relaxing the cell)
        fmax : float, default=0.05
            The threshold for the maximum force to stop the optimization
        cell_relax : bool, default=True
            If True, relax the cell
        model_path : str, optional
            Path to the MACE model file. If None, the default model is used.
        device : {"cuda", "cpu"}, default="cuda"
            Computational device for running simulations, either "cuda" for GPU or "cpu".
        default_dtype : {"float32", "float64"}, default="float32"
            Default data type for the MACE model.
        dispersion : bool, default=True
            Whether to include dispersion correction in the energy calculations.
        """
        self.num_total_optimization = num_total_optimization
        self.num_internal_steps = num_internal_steps
        self.num_cell_steps = num_cell_steps
        self.fmax = fmax
        self.cell_relax = cell_relax
        self.model_path = model_path
        self.device = device
        self.default_dtype = default_dtype
        self.dispersion = dispersion

        # Configure device
        device = self._configure_device(device)

        # Set up calculator
        self.calculator = self._initialize_calculator(
            model_path, device, default_dtype, dispersion
        )

    def _configure_device(self, device: str) -> str:
        if device not in ["cuda", "cpu"]:
            raise ValueError("Device must be either 'cuda' or 'cpu'")
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead")
            device = "cpu"
        return device

    def _initialize_calculator(
        self,
        model_path: Optional[str],
        device: Literal["cuda", "cpu"],
        default_dtype: Literal["float32", "float64"],
        dispersion: bool,
    ):
        model_path = Path(model_path) if model_path else Path(DEFAULT_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"Using model at: {model_path}")
        return mace_mp(
            model=model_path,
            device=device,
            default_dtype=default_dtype,
            dispersion=dispersion,
        )

    def run(self, atoms: Atoms, trajectory_file: Optional[str] = None):
        """Perform geometry optimization.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to optimize
        trajectory_file : str, optional
            Path to the trajectory file. If provided, the trajectory will be saved.
        """
        return optimize_atoms(
            calculator=self.calculator,
            atoms=atoms,
            num_total_optimization=self.num_total_optimization,
            num_internal_steps=self.num_internal_steps,
            num_cell_steps=self.num_cell_steps,
            fmax=self.fmax,
            cell_relax=self.cell_relax,
            trajectory_file=trajectory_file,
        )


def optimize_atoms(
    calculator: Calculator,
    atoms: Atoms,
    num_total_optimization: int = 30,
    num_internal_steps: int = 50,
    num_cell_steps: int = 50,
    fmax: float = 0.05,
    cell_relax: bool = True,
    trajectory_file: Optional[str] = None,
) -> Optional[Atoms]:
    """Perform geometry optimization using the FIRE algorithm.

    Parameters
    ----------
    calculator : Calculator
        ASE calculator for the optimization
    atoms : Atoms
        Atoms object to optimize
    num_total_optimization : int, default=30
        The number of optimization steps including internal and cell relaxation
    num_internal_steps : int, default=50
        The number of internal steps (freezing the cell)
    num_cell_steps : int, default=50
        The number of optimization steps (relaxing the cell)
    fmax : float, default=0.05
        The threshold for the maximum force to stop the optimization
    cell_relax : bool, default=True
        If True, relax the cell
    trajectory_file : str, optional
        Path to the trajectory file. If provided, the trajectory will be saved.

    Returns
    -------
    Optional[Atoms]
        The optimized atoms object
    """
    if trajectory_file is not None:
        trajectory = Trajectory(trajectory_file, "w", atoms)

    opt_atoms = atoms.copy()
    convergence = False

    for _ in range(int(num_total_optimization)):
        opt_atoms = opt_atoms.copy()
        opt_atoms.calc = calculator

        # cell relaxation
        if cell_relax:
            filter = FrechetCellFilter(opt_atoms)  # pylint: disable=redefined-builtin
            optimizer = FIRE(filter)
            convergence = optimizer.run(fmax=fmax, steps=num_cell_steps)
            opt_atoms.wrap()
            if trajectory_file is not None:
                optimizer.attach(trajectory.write, interval=1)
            convergence = optimizer.run(fmax=fmax, steps=num_internal_steps)
            if convergence:
                break

        # internal relaxation
        optimizer = FIRE(opt_atoms)
        convergence = optimizer.run(fmax=fmax, steps=num_internal_steps)
        if trajectory_file is not None:
            optimizer.attach(trajectory.write, interval=1)
        if convergence and not cell_relax:
            break

        # fail if the forces are too large
        forces = filter.get_forces()
        _fmax = np.sqrt((forces**2).sum(axis=1).max())
        if _fmax > 1000:
            return None

    if not convergence:
        return None
    return opt_atoms
