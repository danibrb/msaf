"""
Entry point for the KMC epitaxial growth simulation of a 2D Ag monolayer.
"""

import numpy as np
import matplotlib.pyplot as plt
import time as _time

import config
from lattice import initialise_lattice, coverage
from numerics import set_numba_seed
from kmc import run
from output import (
    ensure_output_dir,
    report_initial_state,
    save_snapshot_mosaic,
    save_coverage_plots,
)
from visualization import plot_lattice


# 0. Prepare output folder

ensure_output_dir()


# 1. Random generators
#    - NumPy Generator (rng)   : used for the Python-level lattice initialisation
#    - Numba internal RNG      : seeded separately via set_numba_seed()
#      (Numba maintains its own Mersenne-Twister, independent of numpy's)

rng = np.random.default_rng(config.SEED)
set_numba_seed(config.SEED)


# 2. Initialise lattice and report

lattice = initialise_lattice(rng)
report_initial_state(lattice)


# 3. Save initial snapshot

fig_init = plot_lattice(
    lattice,
    title=f"t = 0   N = {config.N_INIT} atoms   θ = {coverage(lattice):.4f}",
)
_path_init = config.OUTPUT_DIR / f"snapshot_initial_T_{config.T}.png"
fig_init.savefig(_path_init, dpi=150, bbox_inches="tight")
plt.close(fig_init)
print(f"  Saved : {_path_init}\n")


# 4. Warm up Numba (first call triggers JIT compilation, not counted in timing)

print("  Warming up Numba JIT compiler (first-call compilation) ...")

_t0 = _time.perf_counter()
_dummy = initialise_lattice(np.random.default_rng(0))
run(_dummy, n_deposit=1, snapshot_every=1)   # single-step warm-up
_t1 = _time.perf_counter()
print(f"  JIT warm-up complete ({_t1 - _t0:.1f} s)\n")


# 5. Run full KMC simulation

_t0 = _time.perf_counter()
results = run(lattice, n_deposit=config.N_DEPOSIT, snapshot_every=config.SNAPSHOT_EVERY)
_t1 = _time.perf_counter()
print(f"  Wall-clock time : {_t1 - _t0:.2f} s\n")


# 6. Save final snapshot

fig_final = plot_lattice(
    lattice,
    title=(
        f"Final state   N_dep = {config.N_DEPOSIT}"
        f"   θ = {coverage(lattice):.4f}"
    ),
)
_path_final = config.OUTPUT_DIR / f"snapshot_final_T_{config.T}.png"
fig_final.savefig(_path_final, dpi=150, bbox_inches="tight")
plt.close(fig_final)
print(f"  Saved : {_path_final}")


# 7. Save snapshot mosaic and coverage plots

save_snapshot_mosaic(results["snapshots"], results["n_deposited"], results["coverage"])
save_coverage_plots(results)

print(f"\n  All output written to {config.OUTPUT_DIR}")
