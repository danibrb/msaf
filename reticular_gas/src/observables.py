"""
observables.py
--------------
Python-level statistical analysis of recorded Monte Carlo trajectories.

These functions operate on plain Python lists or numpy arrays of order-parameter
values recorded during a simulation run.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence

__all__ = ["equilibration_mean", "autocorrelation_time"]


def equilibration_mean(
    order_param_history: Sequence[float],
    equilibration_fraction: float = 0.5,
) -> tuple[float, float]:
    """
    Estimate ⟨|P|⟩ from the production part of the trajectory.

    The first `equilibration_fraction` of the recorded values are discarded
    as equilibration (thermalisation) data.  The mean and standard deviation
    are computed over |P| of the remaining samples.

    Parameters
    ----------
    order_param_history   : recorded P values (signed), one per sampling event.
    equilibration_fraction: fraction in [0, 1) of initial samples to discard.

    Returns
    -------
    mean : time-average of |P| over the production run.
    std  : sample standard deviation of |P|.
    """
    data = np.asarray(order_param_history, dtype=float)
    start_idx = int(len(data) * equilibration_fraction)
    production = np.abs(data[start_idx:])
    return float(np.mean(production)), float(np.std(production, ddof=1))


def autocorrelation_time(
    order_param_history: Sequence[float],
    max_lag_fraction: float = 0.5,
) -> float:
    """
    Estimate the integrated autocorrelation time τ_int of the order parameter.

    Uses the self-consistent window estimator:

        τ_int ≈ 0.5 + Σ_{k=1}^{K} Γ(k) / Γ(0)

    where K is the first lag at which the normalised autocorrelation Γ(k)/Γ(0)
    drops below zero (automatic windowing).  τ_int provides the statistical
    inefficiency: the variance of the mean is inflated by a factor 2 τ_int
    relative to independent sampling.

    Reference: Madras, N. & Sokal, A. D. (1988). J. Stat. Phys., 50(1–2), 109–186.

    Parameters
    ----------
    order_param_history : recorded P values.
    max_lag_fraction    : upper limit on lag as a fraction of the series length
                          (guards against noise at large lags).

    Returns
    -------
    tau_int : integrated autocorrelation time in units of sampling intervals.
              Returns 0.0 if the signal has zero variance.
    """
    data = np.asarray(order_param_history, dtype=float)
    data = data - np.mean(data)
    n = len(data)
    gamma_0 = float(np.dot(data, data)) / n
    if gamma_0 == 0.0:
        return 0.0

    tau_int = 0.5
    max_lag = int(n * max_lag_fraction)

    for k in range(1, max_lag):
        gamma_k = float(np.dot(data[:n - k], data[k:])) / (n - k)
        normalised = gamma_k / gamma_0
        if normalised <= 0.0:
            break
        tau_int += normalised

    return tau_int
