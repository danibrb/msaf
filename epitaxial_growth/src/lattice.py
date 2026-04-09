"""
lattice.py
----------
Lattice representation and geometric utilities for the KMC epitaxial growth
simulation on a 2D square lattice.

The lattice is an L x L NumPy array of dtype int8:
    0  →  empty site
    1  →  occupied site (adatom present)

Periodic boundary conditions (PBC) are enforced throughout, consistent with
the standard treatment of the 2D lattice gas model (see lecture notes, §"Modello
a gas reticolare in 2 dimensioni").

Nearest-neighbor topology: 4 sites (von Neumann neighbourhood).
    offsets: (-1,0), (+1,0), (0,-1), (0,+1)
"""

from __future__ import annotations

import numpy as np

import config

# Four cardinal offsets for the square lattice (von Neumann neighbourhood)
_NN_OFFSETS: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]



# Lattice creation


def create_empty_lattice() -> np.ndarray:
    """Return an L × L lattice with all sites empty.

    Returns
    -------
    np.ndarray
        Integer array of shape (L, L) with dtype int8, initialised to zero.
    """
    return np.zeros((config.L, config.L), dtype=np.int8)


def place_particles(
    lattice: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly place *n* adatoms on vacant sites.

    Each particle is positioned by rejection sampling: candidate (row, col)
    pairs are drawn uniformly and accepted only when the site is empty,
    thereby avoiding double occupation.

    Parameters
    ----------
    lattice : np.ndarray
        The L × L occupation array (modified in place).
    n : int
        Number of adatoms to deposit.
    rng : np.random.Generator
        NumPy random generator (seeded externally for reproducibility).

    Returns
    -------
    np.ndarray
        The modified lattice (same object, returned for convenience).

    Raises
    ------
    ValueError
        If the lattice does not have enough vacant sites to accommodate *n*
        additional particles.
    """
    L = config.L
    n_vacant = int(np.sum(lattice == 0))
    if n > n_vacant:
        raise ValueError(
            f"Cannot place {n} particles: only {n_vacant} vacant sites available."
        )

    placed = 0
    while placed < n:
        row = rng.integers(0, L)
        col = rng.integers(0, L)
        if lattice[row, col] == 0:
            lattice[row, col] = 1
            placed += 1

    return lattice


def initialise_lattice(rng: np.random.Generator) -> np.ndarray:
    """Create a lattice and place ``config.N_INIT`` particles.

    This is the canonical entry point used by the KMC driver.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    np.ndarray
        Occupied L × L lattice.
    """
    lattice = create_empty_lattice()
    place_particles(lattice, config.N_INIT, rng)
    return lattice



# Nearest-neighbour geometry


def get_neighbors(row: int, col: int) -> list[tuple[int, int]]:
    """Return the four nearest-neighbour site indices with PBC.

    Parameters
    ----------
    row, col : int
        Site coordinates on the L × L lattice.

    Returns
    -------
    list of (int, int)
        The four neighbour coordinates, each wrapped modulo L.
    """
    L = config.L
    return [
        ((row + dr) % L, (col + dc) % L)
        for dr, dc in _NN_OFFSETS
    ]


def count_neighbors(lattice: np.ndarray, row: int, col: int) -> int:
    """Count how many of the four nearest-neighbour sites are occupied.

    This quantity, denoted n_pv in the lecture notes, enters directly into
    the effective energy barrier:

        E_i = E0 + n_pv * Eb

    and therefore into the Arrhenius hopping rate (TST, Eq. 56 of the notes).

    Parameters
    ----------
    lattice : np.ndarray
        Current occupation array.
    row, col : int
        Coordinates of the site under examination.

    Returns
    -------
    int
        Number of occupied nearest-neighbour sites, in [0, 4].
    """
    return int(sum(lattice[r, c] for r, c in get_neighbors(row, col)))


def neighbor_map(lattice: np.ndarray) -> np.ndarray:
    """Compute the nearest-neighbour count for every site simultaneously.

    Uses ``np.roll`` to shift the lattice along each axis, accumulating
    contributions from all four cardinal directions. This vectorised
    approach is preferable for bulk diagnostics or visualisation; the
    per-site ``count_neighbors`` function should be used inside the KMC
    loop for individual updates.

    Parameters
    ----------
    lattice : np.ndarray
        Current occupation array of shape (L, L).

    Returns
    -------
    np.ndarray
        Integer array of shape (L, L) where entry [i, j] gives the number
        of occupied nearest neighbours of site (i, j).
    """
    nn = np.zeros_like(lattice, dtype=np.int8)
    for dr, dc in _NN_OFFSETS:
        nn += np.roll(lattice, shift=(-dr, -dc), axis=(0, 1))
    return nn



# Occupied-site catalogue


def get_occupied_sites(lattice: np.ndarray) -> list[tuple[int, int]]:
    """Return a list of (row, col) coordinates for all occupied sites.

    Parameters
    ----------
    lattice : np.ndarray
        Current occupation array.

    Returns
    -------
    list of (int, int)
    """
    rows, cols = np.where(lattice == 1)
    return list(zip(rows.tolist(), cols.tolist()))


def coverage(lattice: np.ndarray) -> float:
    """Return the fractional surface coverage θ = N_p / L².

    Parameters
    ----------
    lattice : np.ndarray
        Current occupation array.

    Returns
    -------
    float
        Coverage in [0, 1].
    """
    return float(np.sum(lattice)) / (config.L ** 2)
