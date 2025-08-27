import numpy as np
from ase.atoms import Atoms


def add_molecule(gas: Atoms, rotate: bool = True, translate: tuple = None) -> Atoms:
    """Add a molecule to the simulation cell

    Parameters
    ----------
    gas : Atoms
        The gas molecule to add
    rotate : bool, optional
        If True, rotate the molecule randomly, by default True
    translate : tuple, optional
        The translation of the molecule, by default None

    Returns
    -------
    Atoms
        The gas molecule added to the simulation cell

    Raises
    ------
    ValueError
        If the translate is not a 3-tuple, raise an error

    Examples
    --------
    >>> from ml_mc.utils import molecule, add_gas
    >>> gas = molecule('H2O')
    >>> gas = add_gas(gas, rotate=True, translate=(0, 0, 0))
    """
    gas = gas.copy()
    if rotate:
        angle = np.random.rand() * 360
        axis = np.random.rand(3)
        gas.rotate(v=axis, a=angle)
    if translate is not None:
        if len(translate) != 3:
            raise ValueError("translate must be a 3-tuple")
        gas.translate(translate)
    return gas
