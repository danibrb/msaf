"""
montecarlo.py
-------------
Numba-compiled kernels for the Metropolis Monte Carlo algorithm.

All performance-critical functions are decorated with @njit (no-Python mode).
The global DIRECTIONS array is captured by Numba at first compilation and
treated as a compile-time constant; it must not be reassigned after import.

References
----------
Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., &
    Teller, E. (1953). Equation of state calculations by fast computing
    machines. J. Chem. Phys., 21(6), 1087–1092.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from constants import DIRECTIONS, SEED
from initialization import initialize_lattice

__all__ = [
    "seed_numba_rng",
    "compute_delta_energy",
    "mc_step",
    "compute_order_parameter",
    "warmup_jit",
]


@njit
def seed_numba_rng(seed: int) -> None:
    """
    Seed Numba's internal Mersenne-Twister RNG.

    Numba maintains an RNG state that is entirely independent of NumPy's.
    This function must be called once before the simulation loop to ensure
    reproducible trajectories.
    """
    np.random.seed(seed)


@njit
def compute_delta_energy(
    px: int,
    py: int,
    nx: int,
    ny: int,
    lattice: np.ndarray,
    L: int,
) -> int:
    """
    Compute ΔE = E_after − E_before for a single-particle hop (px,py) → (nx,ny).

    The Hamiltonian is H = J Σ_{⟨l,l'⟩} n_l n_{l'} with J > 0 (repulsive).
    Only the nearest-neighbour bonds that involve either (px,py) or (nx,ny)
    can change upon the hop; all other pairs are unaffected.

    Counting rules (no lattice mutation required):
    - energy_before : occupied neighbours of (px,py).
      (nx,ny) is guaranteed empty by the caller → it contributes 0 automatically.
    - energy_after  : occupied neighbours of (nx,ny), EXCLUDING (px,py),
      which becomes empty after the hop.

    Parameters
    ----------
    px, py : coordinates of the departure site (currently occupied).
    nx, ny : coordinates of the arrival site (currently empty).
    lattice : bool occupation array.
    L       : lattice linear size (for periodic boundary conditions).

    Returns
    -------
    ΔE (in units of J) as a signed integer.
    """
    energy_before = 0
    energy_after = 0

    for d in range(4):
        dx = DIRECTIONS[d, 0]
        dy = DIRECTIONS[d, 1]

        # --- contribution from departure site neighbours ---
        vx = (px + dx) % L
        vy = (py + dy) % L
        # (nx,ny) is empty → lattice[nx,ny] == False → automatically excluded
        if lattice[vx, vy]:
            energy_before += 1

        # --- contribution from arrival site neighbours ---
        wx = (nx + dx) % L
        wy = (ny + dy) % L
        # (px,py) will be vacated → exclude it explicitly
        if lattice[wx, wy] and not (wx == px and wy == py):
            energy_after += 1

    return energy_after - energy_before


@njit
def mc_step(
    coords: np.ndarray,
    lattice: np.ndarray,
    L: int,
    N: int,
    betaj: float,
) -> tuple[int, int]:
    """
    Perform one Monte Carlo step: N attempted single-particle hops.

    Each attempt proceeds as follows (Metropolis algorithm):
      1. Draw a particle index uniformly from [0, N).
      2. Draw a direction uniformly from the four nearest-neighbour directions.
      3. Reject immediately if the target site is occupied (hard-core exclusion).
      4. Compute ΔE via compute_delta_energy.
      5. Accept with probability min(1, exp(−βJ · ΔE)).

    The RNG state is NOT re-seeded here; call seed_numba_rng once before the loop.

    Parameters
    ----------
    coords  : (N, 2) int32 particle coordinate array (modified in place).
    lattice : (L, L) bool occupation array (modified in place).
    L       : lattice linear size.
    N       : number of particles.
    betaj   : reduced inverse temperature βJ.

    Returns
    -------
    accepted : number of accepted moves in this step.
    rejected : number of rejected moves in this step.
    """
    accepted = 0
    rejected = 0

    for _ in range(N):
        # --- select a random particle and a random direction ---
        particle_idx = np.random.randint(0, N)
        px = coords[particle_idx, 0]
        py = coords[particle_idx, 1]

        direction = np.random.randint(0, 4)
        dx = DIRECTIONS[direction, 0]
        dy = DIRECTIONS[direction, 1]
        nx = (px + dx) % L
        ny = (py + dy) % L

        # hard-core exclusion: reject if target site is occupied
        if lattice[nx, ny]:
            rejected += 1
            continue

        delta_e = compute_delta_energy(px, py, nx, ny, lattice, L)

        # Metropolis criterion: always accept if ΔE ≤ 0
        if delta_e > 0 and np.random.rand() > np.exp(-betaj * delta_e):
            rejected += 1
            continue

        # accept: update occupation array and coordinate table
        lattice[px, py] = False
        lattice[nx, ny] = True
        coords[particle_idx, 0] = nx
        coords[particle_idx, 1] = ny
        accepted += 1

    return accepted, rejected


@njit
def compute_order_parameter(coords: np.ndarray, N: int) -> float:
    """
    Compute the c(2×2) order parameter P = (N_A − N_B) / N.

    Sublattice A: sites (x, y) with (x + y) even.
    Sublattice B: sites (x, y) with (x + y) odd.

    In the fully ordered c(2×2) phase, particles preferentially occupy one
    sublattice → |P| → 1.  In the disordered phase → |P| → 0.

    Reference: lecture notes, Eq. (43).
    """
    n_A = 0
    n_B = 0
    for i in range(N):
        if (coords[i, 0] + coords[i, 1]) % 2 == 0:
            n_A += 1
        else:
            n_B += 1
    return (n_A - n_B) / N


def warmup_jit(seed: int = SEED) -> None:
    """
    Pre-compile all Numba kernels on a minimal 4×4 test lattice.

    Invoke once at program start to pay the JIT compilation cost upfront
    and separate it from the actual simulation timing.
    """
    L_warm = 4
    N_warm = 4
    coords_warm, lattice_warm = initialize_lattice(L_warm, N_warm, seed)
    seed_numba_rng(seed)
    mc_step(coords_warm, lattice_warm, L_warm, N_warm, 1.0)
    compute_order_parameter(coords_warm, N_warm)
    # also trigger compilation of compute_delta_energy with valid arguments
    compute_delta_energy(0, 0, 1, 0, lattice_warm, L_warm)
