"""
runner.py
---------
High-level simulation orchestration.

Provides two public functions:
  - run_simulation        : single (L, θ, βJ) canonical MC run.
  - run_multi_parameter_scan : sequential sweep over (βJ, θ) grids.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import config
from constants import SEED
from initialization import initialize_lattice
from montecarlo import mc_step, compute_order_parameter, seed_numba_rng
from observables import equilibration_mean, autocorrelation_time
from visualization import plot_lattice, plot_order_parameter, plot_heatmap

__all__ = ["run_simulation", "run_multi_parameter_scan"]


def run_simulation(
    L: int,
    theta: float,
    betaj: float,
    num_steps: int,
    sampling_interval: int,
    seed: int = SEED,
    show_lattice: bool = True,
    show_order_parameter: bool = True,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, list[int], list[float]]:
    """
    Run a single canonical-ensemble MC simulation of the 2D lattice gas.

    The simulation evolves N = round(θ L²) hard-core particles on a
    L×L periodic square lattice with nearest-neighbour repulsion J.
    Dynamics are governed by the Metropolis algorithm (see montecarlo.py).

    Parameters
    ----------
    L                    : lattice linear size.
    theta                : filling fraction θ = N / L².
    betaj                : reduced inverse temperature βJ = J / (k_B T).
    num_steps            : total Monte Carlo steps.
    sampling_interval    : steps between P measurements.
    seed                 : master RNG seed (initialisation + Numba).
    show_lattice         : display the final lattice snapshot.
    show_order_parameter : display the P(t) trajectory plot.
    output_dir           : if provided, figures are saved there as PNG files.
    verbose              : print progress and summary statistics to stdout.

    Returns
    -------
    coords_final        : (N, 2) int32 array of final particle positions.
    steps_recorded      : list of MC step indices at which P was sampled.
    order_param_history : list of corresponding signed P values.
    """
    N = round(theta * L * L)

    if verbose:
        print(f"  L={L}, θ={theta:.3f}, βJ={betaj:.4f}, N={N}, steps={num_steps}")

    t_start = time.perf_counter()

    coords, lattice = initialize_lattice(L, N, seed)
    seed_numba_rng(seed)   # seed Numba's RNG once before the loop

    # record initial state
    steps_recorded: list[int] = [0]
    order_param_history: list[float] = [float(compute_order_parameter(coords, N))]

    total_accepted = 0
    total_rejected = 0

    for step in range(1, num_steps + 1):
        accepted, rejected = mc_step(coords, lattice, L, N, betaj)
        total_accepted += accepted
        total_rejected += rejected

        if step % sampling_interval == 0:
            p = float(compute_order_parameter(coords, N))
            steps_recorded.append(step)
            order_param_history.append(p)

    elapsed = time.perf_counter() - t_start
    acceptance_rate = total_accepted / max(total_accepted + total_rejected, 1) * 100.0
    p_mean, p_std = equilibration_mean(order_param_history)
    tau = autocorrelation_time(order_param_history)

    if verbose:
        print(f"  Elapsed:         {elapsed:.2f} s")
        print(f"  Acceptance rate: {acceptance_rate:.1f} %")
        print(f"  ⟨|P|⟩ = {p_mean:.4f} ± {p_std:.4f}   (τ_int ≈ {tau:.1f} MCS)")

    # ── figures ────────────────────────────────────────────────────────────────
    tag = f"L{L}_th{theta:.2f}_bj{betaj:.4f}"
    out = Path(output_dir) if output_dir is not None else None

    if show_lattice:
        lattice_path = (out / f"lattice_{tag}.png") if out else None
        plot_lattice(coords, L, "Final Configuration", theta, betaj, lattice_path)

    if show_order_parameter:
        op_path = (out / f"order_param_{tag}.png") if out else None
        plot_order_parameter(
            steps_recorded, order_param_history, L, theta, betaj, op_path
        )

    return coords, steps_recorded, order_param_history


def run_multi_parameter_scan(
    L: int,
    betaj_values: list[float],
    theta_values: list[float],
    num_steps: int,
    sampling_interval: int = 20,
    output_dir: Path | None = None,
) -> np.ndarray:
    """
    Sequential sweep over the (βJ, θ) parameter grid.

    Each (βJ, θ) combination is simulated independently with a distinct
    RNG seed derived from the master seed.  The production-run mean ⟨|P|⟩
    is stored in a 2D result array and visualised as a heatmap.

    Parameters
    ----------
    L                : lattice linear size (identical for all runs).
    betaj_values     : list of βJ values (heatmap rows, ascending index).
    theta_values     : list of θ values  (heatmap columns).
    num_steps        : MC steps per run.
    sampling_interval: steps between P measurements.
    output_dir       : optional directory for saving figures.

    Returns
    -------
    results : float array of shape (len(betaj_values), len(theta_values))
              containing ⟨|P|⟩ for each parameter combination.
    """
    n_betaj = len(betaj_values)
    n_theta = len(theta_values)
    results = np.zeros((n_betaj, n_theta), dtype=float)
    total_runs = n_betaj * n_theta
    run_counter = 0

    for i, betaj in enumerate(betaj_values):
        for j, theta in enumerate(theta_values):
            run_counter += 1
            run_seed = SEED + run_counter   # unique seed per parameter point

            print(f"  [{run_counter:3d}/{total_runs}]  βJ={betaj:.4f},  θ={theta:.3f}")

            _, _, op_history = run_simulation(
                L=L,
                theta=theta,
                betaj=betaj,
                num_steps=num_steps,
                sampling_interval=sampling_interval,
                seed=run_seed,
                show_lattice=False,
                show_order_parameter=False,
                output_dir=None,
                verbose=False,
            )

            p_mean, p_std = equilibration_mean(op_history)
            results[i, j] = p_mean
            print(f"           ⟨|P|⟩ = {p_mean:.4f} ± {p_std:.4f}")

    out = Path(output_dir) if output_dir is not None else None
    heatmap_path = (out / f"heatmap_L{L}.png") if out else None
    plot_heatmap(results, betaj_values, theta_values, L, heatmap_path)

    return results
