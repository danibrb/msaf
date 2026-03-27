"""
constants.py
------------
Physical and numerical constants shared across all modules.
"""
from __future__ import annotations

import numpy as np

__all__ = ["SEED", "DIRECTIONS", "BETAJ_CRITICAL"]

# Master RNG seed — change here for a different realisation
SEED: int = 1_234_567_890

# Nearest-neighbour displacement vectors on the 2D square lattice (periodic BC)
# Rows: [−x, +x, −y, +y]
DIRECTIONS: np.ndarray = np.array(
    [[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32
)

# Exact critical inverse temperature at θ = 0.5 (Onsager mapping):
# βJ_c = 2 ln(1 + √2) ≈ 1.7627  (cf. lecture notes, Eq. 42)
BETAJ_CRITICAL: float = 2.0 * np.log(1.0 + np.sqrt(2.0))
