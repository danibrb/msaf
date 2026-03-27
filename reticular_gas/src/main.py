"""
main.py
-------
Entry point for the 2D Lattice Gas Monte Carlo simulation.

Physical model
--------------
A square lattice of L×L sites with periodic boundary conditions.
Each site carries an occupation variable n_l ∈ {0, 1} (hard-core exclusion).
Nearest-neighbour pairs interact via repulsive potential of strength J > 0:

    H = J Σ_{⟨l,l'⟩} n_l n_{l'}

At filling θ = 0.5 the system undergoes a continuous order–disorder transition
of the c(2×2) type at the exact critical temperature

    βJ_c = 2 ln(1 + √2) ≈ 1.7627          (lecture notes, Eq. 42)

The order parameter is P = (N_A − N_B) / N, where A and B denote the two
interpenetrating sublattices of the square lattice.

Usage
-----
    python main.py

References
----------
Metropolis, N., et al. (1953). Equation of State Calculations by Fast Computing
    Machines. J. Chem. Phys., 21(6), 1087–1092.
Binder, K. & Heermann, D. W. (2010). Monte Carlo Simulation in Statistical
    Physics. Springer, 5th ed.
"""
from __future__ import annotations

import time
from pathlib import Path

import config
from montecarlo import warmup_jit
from runner import run_simulation, run_multi_parameter_scan

OUTPUT_DIR = Path(config.OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    # ── 1. Pre-compile Numba kernels ──────────────────────────────────────────
    print("Warming up Numba JIT compilation (runs once per session) …")
    t_jit = time.perf_counter()
    warmup_jit()
    print(f"JIT compilation completed in {time.perf_counter() - t_jit:.2f} s\n")

    # ── 2. Single simulation ──────────────────────────────────────────────────
    print("=" * 60)
    print("SINGLE SIMULATION")
    print("=" * 60)

    run_simulation(
        L=config.L,
        theta=config.THETA,
        betaj=config.BETAJ,
        num_steps=config.NUM_STEPS,
        sampling_interval=config.SAMPLING_INTERVAL,
        show_lattice=True,
        show_order_parameter=True,
        output_dir=OUTPUT_DIR,
        verbose=True,
    )

    # ── 3. Parameter-space scan ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MULTI-PARAMETER SCAN")
    print("=" * 60)

    run_multi_parameter_scan(
        L=config.L_SCAN,
        betaj_values=config.BETAJ_SCAN,
        theta_values=config.THETA_SCAN,
        num_steps=config.NUM_STEPS_SCAN,
        sampling_interval=config.SAMPLING_INTERVAL,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
