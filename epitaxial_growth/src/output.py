"""
I/O and reporting utilities for the KMC epitaxial growth simulation.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

import config
from lattice import count_neighbors, get_occupied_sites, coverage
from rates import HOPPING_RATES



# Directory management


def ensure_output_dir() -> None:
    """
    Create output directory if absent.
    """
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory : {config.OUTPUT_DIR}")



# Initial-state diagnostics


def report_initial_state(lattice: np.ndarray) -> None:
    """
    Print coordinates, n_pv counts, and hopping rates for every initial atom.
    """
    occupied = get_occupied_sites(lattice)
    theta    = coverage(lattice)

    print("=" * 62)
    print("  KMC epitaxial growth  ")
    print("=" * 62)
    print(f"  Lattice size    :  {config.L} x {config.L} = {config.L**2} sites")
    print(f"  Temperature     :  {config.T:.0f} K")
    print(f"  E0 / Eb         :  {config.E0} / {config.EB} eV")
    print(f"  Nu0              :  {config.NU0:.2e} s^-1")
    print(f"  Flux F          :  {config.F:.4f} ML s^-1")
    print(f"  N_DEPOSIT       :  {config.N_DEPOSIT}")
    print(f"  Initial atoms   :  {len(occupied)}")
    print(f"  Coverage Theta     :  {theta:.6f}")
    print("-" * 62)
    print(f"  {'Site (row, col)':<24} {'n_pv':>6}  {'rate [s^-1]':>14}")
    print("-" * 62)
    for row, col in occupied:
        npv  = count_neighbors(lattice, row, col)
        rate = HOPPING_RATES[npv]
        print(f"  ({row:>3d}, {col:>3d}){'':<18} {npv:>6d}  {rate:>14.4e}")
    print("=" * 62)



# Snapshot mosaic


def save_snapshot_mosaic(
    snapshots:  list[np.ndarray],
    dep_counts: np.ndarray,
    coverages:  np.ndarray,
) -> None:
    """
    Tile all recorded checkpoint snapshots into a single PNG and save it.
    """
    n = len(snapshots)
    if n == 0:
        return

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    cmap  = mcolors.ListedColormap(["#f7f7f7", "#2166ac"])

    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.35, wspace=0.15)

    for idx, (snap, nd, theta) in enumerate(zip(snapshots, dep_counts, coverages)):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ax.imshow(snap, cmap=cmap, vmin=0, vmax=1,
                  origin="lower", interpolation="nearest")
        ax.set_title(f"N={nd:d}   θ={theta:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n, nrows * ncols):
        fig.add_subplot(gs[idx // ncols, idx % ncols]).set_visible(False)

    fig.suptitle(
        f"Ag/Ag(100) KMC  —  L={config.L}, T={config.T:.0f} K, "
        f"E0={config.E0} eV, Eb={config.EB} eV",
        fontsize=11, y=1.01,
    )

    path = config.OUTPUT_DIR / f"snapshots_mosaic_T_{config.T}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {path}")



# Coverage plots


def save_coverage_plots(results: dict) -> None:
    """
    Write coverage-vs-time and coverage-vs-deposited-atoms plots to disk.
    """
    # coverage vs. simulation time
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(results["time"], results["coverage"], color="#2166ac", linewidth=1.5)
    ax.set_xlabel("Simulation time  [s]")
    ax.set_ylabel(r"Coverage  $\theta$")
    ax.set_title("Surface coverage vs. simulation time")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = config.OUTPUT_DIR / "coverage_vs_time.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {path}")

    # coverage vs. deposited atoms
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(results["n_deposited"], results["coverage"],
            color="#d6604d", linewidth=1.5)
    ax.set_xlabel("Deposited atoms  $N$")
    ax.set_ylabel(r"Coverage  $\theta = N_{\rm tot} / L^2$")
    ax.set_title("Surface coverage vs. deposited atoms")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = config.OUTPUT_DIR / "coverage_vs_deposited.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {path}")
