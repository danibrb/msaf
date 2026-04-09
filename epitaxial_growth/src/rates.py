"""
rates.py
--------
Computation of KMC event rates for the epitaxial growth simulation.

The rate model follows Transition State Theory (TST), as derived in the
lecture notes (Appendix, Eq. 101):

    r_i = nu_0 * exp(-E_i / k_B T)

The effective barrier for a diffusing adatom with n_pv occupied nearest
neighbours in its current site is (Eq. 61 of the notes):

    E_i = E0 + n_pv * Eb

Atoms with n_pv >= 1 neighbours have higher barriers; at low T these
processes become negligible, recovering the DDA limit.

Events are partitioned into classes q = 0, 1, 2, 3, 4 (following Eq. 62-63):
    q = 0..3 → diffusion of an adatom with q occupied neighbours
    q = 4    → deposition (flux F, weight = L² * F)

Within each diffusion class, all events have the same weight, so class
selection followed by uniform draw within the class is exact and efficient
(see §"Monte Carlo a tempo continuo" of the lecture notes).
"""

from __future__ import annotations

import numpy as np

import config
from lattice import count_neighbors, get_neighbors



# Single-event rate


def hopping_rate(n_pv: int) -> float:
    """Arrhenius hopping rate for an adatom with *n_pv* occupied neighbours.

    Parameters
    ----------
    n_pv : int
        Number of occupied nearest-neighbour sites at the departure site.
        Must be in {0, 1, 2, 3, 4}.

    Returns
    -------
    float
        Rate  r = nu_0 * exp(-(E0 + n_pv * Eb) / k_B T)  [events / s].
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
    """Compute the total weight of each process class.

    Each diffusion class q contains n_class[q] mobile atoms (those with
    exactly q occupied neighbours).  Each such atom has 4 - q possible
    target directions (moves to occupied sites are rejected a priori).

    Class weights follow Eq. 62 of the lecture notes:
        p_q = n_q * (4 - q) * nu_0 * exp(-(E0 + q*Eb) / k_B T)

    plus the deposition class (q = 4):
        p_4 = L² * F
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
