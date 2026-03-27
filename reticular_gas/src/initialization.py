"""
initialization.py
-----------------
Random placement of N non-overlapping particles on a L×L square lattice.
"""
from __future__ import annotations

import numpy as np
from constants import SEED

__all__ = ["initialize_lattice"]


def initialize_lattice(
    L: int, N: int, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly distribute N particles over a L×L square lattice.

    Parameters
    ----------
    L    : lattice linear size.
    N    : number of particles (must satisfy N ≤ L²).
    seed : seed for the internal RandomState; ensures reproducibility.

    Returns
    -------
    coords  : int32 array of shape (N, 2); coords[i] = [x_i, y_i].
    lattice : bool array of shape (L, L); lattice[x, y] == True iff occupied.

    Notes
    -----
    A Fisher–Yates shuffle of the flattened site index array guarantees
    uniform sampling without replacement in O(L²) time.
    """
    if N > L * L:
        raise ValueError(f"N={N} exceeds the number of sites L²={L * L}.")

    rng = np.random.RandomState(seed)
    lattice = np.zeros((L, L), dtype=np.bool_)
    coords = np.empty((N, 2), dtype=np.int32)

    all_sites = np.arange(L * L, dtype=np.int32)
    rng.shuffle(all_sites)
    occupied_sites = all_sites[:N]

    for i in range(N):
        x = int(occupied_sites[i] // L)
        y = int(occupied_sites[i] % L)
        coords[i, 0] = x
        coords[i, 1] = y
        lattice[x, y] = True

    return coords, lattice
