"""
visualization.py
----------------
Matplotlib-based visualisation routines for lattice configurations,
order-parameter trajectories, and parameter-space heatmaps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

__all__ = ["plot_lattice", "plot_order_parameter", "plot_heatmap"]


def plot_lattice(
    coords: np.ndarray,
    L: int,
    title: str = "Lattice Gas Configuration",
    theta: float = 0.5,
    betaj: float = 3.0,
    save_path: Path | None = None,
) -> None:
    """
    Render the instantaneous lattice configuration.

    Particles on sublattice A (x + y even) are drawn in blue;
    particles on sublattice B (x + y odd) in red;
    empty sites as open circles in light grey.

    Parameters
    ----------
    coords    : (N, 2) particle coordinate array.
    L         : lattice linear size.
    title     : figure title string.
    theta     : filling fraction (for annotation only).
    betaj     : reduced inverse temperature (for annotation only).
    save_path : if provided, the figure is saved to this path.
    """
    occupied = np.zeros((L, L), dtype=bool)
    sites_A_x: list[int] = []
    sites_A_y: list[int] = []
    sites_B_x: list[int] = []
    sites_B_y: list[int] = []

    for i in range(coords.shape[0]):
        x = int(coords[i, 0])
        y = int(coords[i, 1])
        occupied[x, y] = True
        if (x + y) % 2 == 0:
            sites_A_x.append(x)
            sites_A_y.append(y)
        else:
            sites_B_x.append(x)
            sites_B_y.append(y)

    empty_coords = np.argwhere(~occupied)
    empty_x = empty_coords[:, 0]
    empty_y = empty_coords[:, 1]

    # scale marker size with lattice dimension so the plot is readable at any L
    marker_size = max(5.0, (70.0 / L) ** 2 * 10.0 - 5.0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(empty_x, empty_y, s=marker_size, facecolors="none",
               edgecolors="lavender", linewidths=0.8, alpha=0.6, label="Empty")
    ax.scatter(sites_A_x, sites_A_y, s=marker_size, color="dodgerblue",
               edgecolors="royalblue", linewidths=0.5, label="Sublattice A")
    ax.scatter(sites_B_x, sites_B_y, s=marker_size, color="tomato",
               edgecolors="firebrick", linewidths=0.5, label="Sublattice B")

    tick_step = max(1, L // 8)
    ax.set_xticks(range(0, L, tick_step))
    ax.set_yticks(range(0, L, tick_step))
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.5, L - 0.5)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}\nL={L},  θ={theta:.3f},  βJ={betaj:.4f}", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, markerscale=1.4)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)


def plot_order_parameter(
    steps: list[int],
    order_params: list[float],
    L: int,
    theta: float,
    betaj: float,
    save_path: Path | None = None,
) -> None:
    """
    Plot the time evolution of the c(2×2) order parameter P(t).

    Parameters
    ----------
    steps        : list of Monte Carlo step indices at which P was sampled.
    order_params : corresponding P values (signed).
    L, theta, betaj : simulation parameters for the figure title.
    save_path    : optional file path for saving the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, order_params, color="steelblue", linewidth=1.0, alpha=0.85)
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Monte Carlo Step")
    ax.set_ylabel("Order Parameter  P")
    ax.set_title(
        f"Order Parameter Evolution\nL={L},  θ={theta:.3f},  βJ={betaj:.4f}",
        fontsize=10,
    )
    ax.set_ylim(-1.05, 1.05)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)


def plot_heatmap(
    results: np.ndarray,
    betaj_values: list[float],
    theta_values: list[float],
    L: int,
    save_path: Path | None = None,
) -> None:
    """
    Render ⟨|P|⟩ as a colour map over the (θ, βJ) parameter plane.

    Rows correspond to βJ values; columns to θ values.
    The known critical point βJ_c ≈ 1.7627 at θ = 0.5 is annotated with
    a dashed contour line at |P| = 0.5 to guide the eye.

    Parameters
    ----------
    results       : (len(betaj_values), len(theta_values)) array of ⟨|P|⟩.
    betaj_values  : βJ axis values (rows).
    theta_values  : θ axis values (columns).
    L             : lattice size (annotation only).
    save_path     : optional file path for saving.
    """
    fig, ax = plt.subplots(figsize=(9, 7), dpi=100)

    im = ax.imshow(
        results,
        interpolation="nearest",
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("⟨|P|⟩", fontsize=11)

    # add contour overlay to locate the order–disorder boundary visually
    ax.contour(results, levels=[0.5], colors="white", linewidths=1.2,
               linestyles="--", alpha=0.8)

    ax.set_yticks(range(len(betaj_values)))
    ax.set_yticklabels([f"{b:.4f}" for b in betaj_values], fontsize=8)
    ax.set_xticks(range(len(theta_values)))
    ax.set_xticklabels([f"{t:.2f}" for t in theta_values], fontsize=8)
    ax.set_ylabel("βJ  (reduced inverse temperature)", fontsize=10)
    ax.set_xlabel("θ  (filling fraction)", fontsize=10)
    ax.set_title(
        f"⟨|P|⟩ phase diagram — L={L}\n"
        f"(dashed contour: |P| = 0.5)",
        fontsize=11,
    )

    # annotate each cell with its numerical value
    for i in range(len(betaj_values)):
        for j in range(len(theta_values)):
            val = results[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if val > 0.5 else "black",
                fontweight="bold",
            )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)
