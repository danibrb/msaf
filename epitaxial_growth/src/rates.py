"""
Computation of KMC event rates for the epitaxial growth simulation.
"""

from __future__ import annotations

import numpy as np

import config
from lattice import count_neighbors, get_neighbors



# Single-event rate


def hopping_rate(n_pv: int) -> float:
    """
    Hopping rate for an atom with n_pv occupied neighbours.
    """
    barrier = config.E0 + n_pv * config.EB
    return config.NU0 * np.exp(-barrier / (config.KB * config.T))


# Pre-compute the five possible hopping rates (n_pv = 0..4) once at import.
HOPPING_RATES: np.ndarray = np.array(
    [hopping_rate(q) for q in range(5)], dtype=np.float64
)

# Deposition rate per site is F; total weight for the deposition class is:
DEPOSITION_RATE: float = config.L**2 * config.F



# Class-weight table for a given lattice configuration


def compute_class_weights(
    lattice: np.ndarray,
    n_class: list[int],
) -> np.ndarray:
    """
    Compute the total weight of each process class.
    """
    weights = np.empty(5, dtype=np.float64)
    for q in range(4):
        n_free_directions = 4 - q
        weights[q] = n_class[q] * n_free_directions * HOPPING_RATES[q]
    weights[4] = DEPOSITION_RATE
    return weights


def classify_atoms(
    lattice: np.ndarray,
    occupied: list[tuple[int, int]],
) -> tuple[list[int], list[list[tuple[int, int]]]]:
    """
    Partition occupied sites into the four diffusion classes.
    """
    atoms_by_class: list[list[tuple[int, int]]] = [[] for _ in range(4)]
    for row, col in occupied:
        q = count_neighbors(lattice, row, col)
        if q < 4:   # fully surrounded atoms cannot move
            atoms_by_class[q].append((row, col))
    n_class = [len(atoms_by_class[q]) for q in range(4)]
    return n_class, atoms_by_class
