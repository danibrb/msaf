"""
visualization.py
----------------
Plotting utilities for the KMC epitaxial growth simulation.

Provides:
    plot_lattice  – render a single L × L occupation snapshot as an image.
    plot_coverage – plot surface coverage θ(t) over simulation time.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import config


def plot_lattice(
    lattice: np.ndarray,
    title: str = "Lattice snapshot",
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Render the L x L occupation array as a binary image.
    Empty sites are shown in white; occupied sites in dark blue,
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    cmap = mcolors.ListedColormap(["#f0f0f0", "#2166ac"])
    ax.imshow(lattice, cmap=cmap, vmin=0, vmax=1, origin="lower",
              interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x  [sites]")
    ax.set_ylabel("y  [sites]")
    theta = np.sum(lattice) / config.L**2
    ax.set_xlabel(f"x  [sites]   (θ = {theta:.3f})")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_coverage(
    times: list[float],
    coverages: list[float],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot surface coverage θ as a function of simulation time.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, coverages, color="#2166ac", linewidth=1.5)
    ax.set_xlabel("Simulation time  [s]")
    ax.set_ylabel(r"Coverage  $\theta$")
    ax.set_title("Surface coverage vs. time")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig
