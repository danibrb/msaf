"""
Numba-JIT-compiled kernels for the KMC epitaxial growth simulation.
"""

from __future__ import annotations

import numpy as np
from numba import njit



# RNG seeding


@njit(cache=True)
def set_numba_seed(seed: int) -> None:
    """
    Seed Numba's internal random number generator for reproducibility.
    """
    np.random.seed(seed)



# Nearest-neighbour count


@njit(cache=True)
def count_nn(lattice: np.ndarray, r: int, c: int, L: int) -> int:
    """
    Count occupied nearest-neighbour sites of (r, c) with PBC.
    """
    n = 0
    n += lattice[(r - 1) % L, c]
    n += lattice[(r + 1) % L, c]
    n += lattice[r, (c - 1) % L]
    n += lattice[r, (c + 1) % L]
    return n



# Main KMC kernel


@njit(cache=True)
def kmc_step(
    lattice:   np.ndarray,
    occ_rows:  np.ndarray,
    occ_cols:  np.ndarray,
    n_occ:     int,
    L:         int,
    E0:        float,
    EB:        float,
    NU0:       float,
    F:         float,
    KB:        float,
    T:         float,
) -> tuple:
    """
    Execute one rejection-free KMC event.
    """

    
    # 1. Classify atoms into diffusion classes
    
    # class_idx[q, k] = index into occ_rows/occ_cols of the k-th atom in class q
    n_buf = n_occ + 1   # avoid zero-size allocation when n_occ = 0
    n_class   = np.zeros(4, dtype=np.int64)
    class_idx = np.empty((4, n_buf), dtype=np.int64)

    for k in range(n_occ):
        r   = occ_rows[k]
        c   = occ_cols[k]
        npv = count_nn(lattice, r, c, L)
        if npv < 4:                     # fully surrounded → immobile
            class_idx[npv, n_class[npv]] = k
            n_class[npv] += 1

    
    # 2. Compute per-class weights and total rate P(C)
    
    # p_q = n_q * (4−q) * nu0 * exp(−(E0+q*Eb)/(kB T))   (Eq. 62)
    # p_4 = L² * F                                         (Eq. 63)
    weights = np.empty(5)
    kBT = KB * T
    for q in range(4):
        barrier  = E0 + q * EB
        rate_q   = NU0 * np.exp(-barrier / kBT)
        weights[q] = n_class[q] * (4 - q) * rate_q
    weights[4] = L * L * F

    P_total = 0.0
    for q in range(5):
        P_total += weights[q]

    
    # 3. Draw time increment  δt = −ln(u) / P(C)  (Eq. 54)
    
    dt = -np.log(np.random.random()) / P_total

    
    # 4. Select event class by cumulative-weight sampling (Eq. 65)
    
    u      = np.random.random() * P_total
    cumsum = 0.0
    q_star = 4
    for q in range(5):
        cumsum += weights[q]
        if u <= cumsum:
            q_star = q
            break

    
    # 5a. Deposition event
    
    deposited = False
    if q_star == 4:
        while True:
            r = np.random.randint(0, L)
            c = np.random.randint(0, L)
            if lattice[r, c] == 0:
                lattice[r, c]    = 1
                occ_rows[n_occ]  = r
                occ_cols[n_occ]  = c
                n_occ           += 1
                deposited        = True
                break

    
    # 5b. Diffusion event
    
    else:
        nc      = n_class[q_star]
        occ_idx = class_idx[q_star, np.random.randint(0, nc)]
        r       = occ_rows[occ_idx]
        c       = occ_cols[occ_idx]

        # Enumerate free (vacant) nearest-neighbour sites
        free_r = np.empty(4, dtype=np.int64)
        free_c = np.empty(4, dtype=np.int64)
        n_free = 0

        nr = (r - 1) % L
        if lattice[nr, c] == 0:
            free_r[n_free] = nr;  free_c[n_free] = c;  n_free += 1
        nr = (r + 1) % L
        if lattice[nr, c] == 0:
            free_r[n_free] = nr;  free_c[n_free] = c;  n_free += 1
        nc2 = (c - 1) % L
        if lattice[r, nc2] == 0:
            free_r[n_free] = r;   free_c[n_free] = nc2; n_free += 1
        nc2 = (c + 1) % L
        if lattice[r, nc2] == 0:
            free_r[n_free] = r;   free_c[n_free] = nc2; n_free += 1

        if n_free > 0:
            d   = np.random.randint(0, n_free)
            tr  = free_r[d]
            tc  = free_c[d]
            lattice[r,  c ] = 0
            lattice[tr, tc] = 1
            occ_rows[occ_idx] = tr
            occ_cols[occ_idx] = tc

    return dt, deposited, n_occ
