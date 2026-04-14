"""
Lattice representation and geometric utilities for the KMC epitaxial growth
"""

from __future__ import annotations

import numpy as np

import config

# Four cardinal offsets for the square lattice
_NN_OFFSETS: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]



# Lattice creation


def create_empty_lattice() -> np.ndarray:
    """
    Return an L x L lattice with all sites empty.
    """
    return np.zeros((config.L, config.L), dtype=np.int8)


def place_particles(
    lattice: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly place n atoms on vacant sites.
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
    """
    Create a lattice and place n initial particles.
    """
    lattice = create_empty_lattice()
    place_particles(lattice, config.N_INIT, rng)
    return lattice



# Nearest-neighbour geometry


def get_neighbors(row: int, col: int) -> list[tuple[int, int]]:
    """
    Return the four nearest-neighbour site indices with PBC.
    """
    L = config.L
    return [
        ((row + dr) % L, (col + dc) % L)
        for dr, dc in _NN_OFFSETS
    ]


def count_neighbors(lattice: np.ndarray, row: int, col: int) -> int:
    """
    Count how many of the four nearest-neighbour sites are occupied.
    """
    return int(sum(lattice[r, c] for r, c in get_neighbors(row, col)))


def neighbor_map(lattice: np.ndarray) -> np.ndarray:
    """
    Compute the nearest-neighbour count for every site simultaneously.
    """
    nn = np.zeros_like(lattice, dtype=np.int8)
    for dr, dc in _NN_OFFSETS:
        nn += np.roll(lattice, shift=(-dr, -dc), axis=(0, 1))
    return nn



# Occupied-site catalogue


def get_occupied_sites(lattice: np.ndarray) -> list[tuple[int, int]]:
    """
    Return a list of (row, col) coordinates for all occupied sites.
    """
    rows, cols = np.where(lattice == 1)
    return list(zip(rows.tolist(), cols.tolist()))


def coverage(lattice: np.ndarray) -> float:
    """
    Return the fractional surface coverage θ = N_p / L².
    """
    return float(np.sum(lattice)) / (config.L ** 2)
