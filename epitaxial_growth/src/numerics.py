"""
numerics.py
-----------
Numba-JIT-compiled kernels for the KMC epitaxial growth simulation.

All performance-critical operations are collected here so that Numba can
compile the entire hot path — nearest-neighbour count, class classification,
rate computation, event selection, and lattice update — into a single
ahead-of-time compiled function with no Python overhead per KMC step.

Design notes
~~~~~~~~~~~~
* ``cache=True`` writes the compiled binary to ``__pycache__``, so the JIT
  compilation penalty is paid only on the first run.
* Occupied sites are tracked as two parallel int64 arrays ``occ_rows`` and
  ``occ_cols``, pre-allocated to size L² (the theoretical maximum).  The
  live count is carried in the scalar ``n_occ`` returned from each call.
  In-place update (diffusion) and append (deposition) are both O(1).
* All physical parameters are passed as scalars so that Numba can treat
  them as compile-time constants after specialisation.
* ``set_numba_seed`` seeds Numba's own Mersenne-Twister RNG, which is
  independent of NumPy's generator used for the Python-level initialisation.
"""

from __future__ import annotations

import numpy as np
from numba import njit



# RNG seeding


@njit(cache=True)
def set_numba_seed(seed: int) -> None:
    """Seed Numba's internal random number generator for reproducibility."""
    np.random.seed(seed)



# Nearest-neighbour count (inlined into kmc_step, exposed for diagnostics)


@njit(cache=True)
def count_nn(lattice: np.ndarray, r: int, c: int, L: int) -> int:
    """Count occupied nearest-neighbour sites of (r, c) with PBC.

    Parameters
    ----------
    lattice : np.ndarray (int8, shape L×L)
    r, c    : site row and column indices
    L       : lattice linear size

    Returns
    -------
    int in {0, 1, 2, 3, 4}
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
    """Execute one rejection-free KMC event.

    Implements the continuous-time (n-fold way) algorithm described in the
    lecture notes (§"Monte Carlo a tempo continuo"):
        1.  Partition atoms into diffusion classes q = 0..3 by n_pv count.
        2.  Compute class weights {p_q} and total rate P(C).
        3.  Draw δt ~ Exp(1/P(C))  via the inverse-CDF method (Eq. 54).
        4.  Select class q* by weighted sampling (Eq. 65).
        5.  Execute the selected event (hop or deposition).

    The effective diffusion barrier follows Eq. 61:
        E_q = E0 + q * Eb,  q = n_pv ∈ {0,1,2,3}

    Parameters
    ----------
    lattice   : int8 array of shape (L, L), modified in place.
    occ_rows  : int64 array of length L², active slice [0:n_occ].
    occ_cols  : int64 array of length L², active slice [0:n_occ].
    n_occ     : number of currently occupied sites.
    L         : lattice linear size.
    E0, EB    : bare barrier and lateral bond energy  [eV].
    NU0       : attempt frequency  [s⁻¹].
    F         : deposition flux  [ML s⁻¹].
    KB        : Boltzmann constant  [eV K⁻¹].
    T         : temperature  [K].

    Returns
    -------
    (dt, deposited, n_occ) : (float, bool, int)
        dt        – time increment  [s]
        deposited – True if the event was a deposition
        n_occ     – updated occupied-site count
    """

    # ------------------------------------------------------------------
    # 1. Classify atoms into diffusion classes
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Compute per-class weights and total rate P(C)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Draw time increment  δt = −ln(u) / P(C)  (Eq. 54)
    # ------------------------------------------------------------------
    dt = -np.log(np.random.random()) / P_total

    # ------------------------------------------------------------------
    # 4. Select event class by cumulative-weight sampling (Eq. 65)
    # ------------------------------------------------------------------
    u      = np.random.random() * P_total
    cumsum = 0.0
    q_star = 4
    for q in range(5):
        cumsum += weights[q]
        if u <= cumsum:
            q_star = q
            break

    # ------------------------------------------------------------------
    # 5a. Deposition event
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5b. Diffusion event
    # ------------------------------------------------------------------
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
