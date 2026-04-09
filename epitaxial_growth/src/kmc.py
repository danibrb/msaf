"""
kmc.py
------
Python driver for the Numba-JIT-compiled KMC engine.

The computationally heavy work (neighbour counting, class classification,
rate computation, random event selection, and lattice update) is entirely
delegated to ``numerics.kmc_step``, compiled ahead-of-time by Numba.

Occupied-site representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Numba operates on two parallel int64 arrays ``occ_rows`` and ``occ_cols``
(pre-allocated to L² entries) with a scalar counter ``n_occ``.
Deposition appends at index n_occ (O(1)); diffusion updates in-place (O(1)).
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

import config
from lattice import get_occupied_sites, coverage
from numerics import kmc_step


def run(
    lattice:        np.ndarray,
    n_deposit:      int = config.N_DEPOSIT,
    snapshot_every: int = config.SNAPSHOT_EVERY,
) -> dict:
    """Run the KMC deposition loop until *n_deposit* atoms have been deposited.

    Parameters
    ----------
    lattice : np.ndarray
        Initial L x L occupation array (int8, modified in place).
    n_deposit : int
        Number of atoms to deposit via the KMC flux before stopping.
        Atoms placed during initialisation (config.N_INIT) are not counted.
    snapshot_every : int
        Record a lattice snapshot every this many deposition events.

    Returns
    -------
    dict with keys
        'time'        - np.ndarray of simulation times at checkpoints [s]
        'coverage'    - np.ndarray of theta values at checkpoints
        'n_deposited' - np.ndarray of cumulative deposition counts
        'snapshots'   - list of np.ndarray lattice copies
        'n_kmc_steps' - total KMC steps (deposits + diffusions)
    """
    L   = config.L
    E0  = config.E0
    EB  = config.EB
    NU0 = config.NU0
    F   = config.F
    KB  = config.KB
    T   = config.T

    # Build flat occupied-site buffers pre-allocated to max size L^2
    init_sites = get_occupied_sites(lattice)
    n_occ      = len(init_sites)
    occ_rows   = np.empty(L * L, dtype=np.int64)
    occ_cols   = np.empty(L * L, dtype=np.int64)
    for k, (r, c) in enumerate(init_sites):
        occ_rows[k] = r
        occ_cols[k] = c

    t           = 0.0
    n_deposited = 0
    n_kmc_steps = 0

    times:      list[float]      = []
    coverages:  list[float]      = []
    dep_counts: list[int]        = []
    snapshots:  list[np.ndarray] = []

    pbar = tqdm(
        total=n_deposit,
        desc="Depositing",
        unit="atom",
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} atoms"
            " [{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )

    while n_deposited < n_deposit:
        dt, deposited, n_occ = kmc_step(
            lattice, occ_rows, occ_cols, n_occ,
            L, E0, EB, NU0, F, KB, T,
        )
        t           += dt
        n_kmc_steps += 1

        if deposited:
            n_deposited += 1
            pbar.update(1)

            if n_deposited % snapshot_every == 0 or n_deposited == n_deposit:
                theta = coverage(lattice)
                times.append(t)
                coverages.append(theta)
                dep_counts.append(n_deposited)
                snapshots.append(lattice.copy())

    pbar.close()
    print(
        f"  {n_kmc_steps:,} total KMC steps  "
        f"({n_kmc_steps - n_deposited:,} diffusion, {n_deposited:,} deposition)"
    )

    return {
        "time":        np.array(times),
        "coverage":    np.array(coverages),
        "n_deposited": np.array(dep_counts, dtype=np.int64),
        "snapshots":   snapshots,
        "n_kmc_steps": n_kmc_steps,
    }
