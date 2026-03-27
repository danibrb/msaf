"""
config.py
---------
All user-facing simulation parameters in one place.
Edit this file to change the run without touching any other module.
"""
from __future__ import annotations

import numpy as np
from constants import BETAJ_CRITICAL

# ─── Single simulation ────────────────────────────────────────────────────────

L: int = 64                    # lattice linear size (sites: L × L)
THETA: float = 0.5             # filling fraction  θ = N_particles / L²
BETAJ: float = 3.0             # reduced inverse temperature  βJ = J / (k_B T)
NUM_STEPS: int = 5_000         # total Monte Carlo steps (1 step = N attempted moves)
SAMPLING_INTERVAL: int = 20    # steps between successive order-parameter measurements

# ─── Multi-parameter scan ─────────────────────────────────────────────────────

L_SCAN: int = 32
NUM_STEPS_SCAN: int = 5_000

# βJ values to sweep (the critical value ≈ 1.7627 is included explicitly)
BETAJ_SCAN: list[float] = [0.0, 1.0, 1.4, round(BETAJ_CRITICAL, 4),
                            2.0, 2.5, 3.0, 4.0, 6.0, 8.0]

# θ values to sweep (symmetric around 0.5 as expected from particle–hole symmetry)
THETA_SCAN: list[float] = [0.25, 0.35, 0.50, 0.65, 0.75]

# ─── Output ───────────────────────────────────────────────────────────────────

OUTPUT_DIR: str = "output"
