"""
Simulation configuration
"""

import numpy as np

L: int = 8                  # Matrix size
theta: float = .5           # Filling

N: int = int(L * L * theta) # Total particle number

# Reproducibility

seed = 1234567890

rng_state = np.random.default_rng(seed)