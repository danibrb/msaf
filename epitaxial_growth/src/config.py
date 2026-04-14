"""
Physical parameters for kinetic Monte Carlo simulation of epitaxial growth
"""

from pathlib import Path


# Paths

SRC_DIR: Path    = Path(__file__).resolve().parent
OUTPUT_DIR: Path = SRC_DIR.parent / "output"


# Physical constants

KB: float = 8.617_333_262e-5   # Boltzmann constant  [eV / K]


# Lattice geometry

L: int = 100             # Linear size of the L × L square lattice [sites]


# energy parameters  [eV]

E0: float  = 0.4        # Diffusion barrier for an isolated atom
EB: float  = 0.2        # Lateral bond energy increment per nearest neighbour
NU0: float = 1.0e12     # Attempt frequency prefactor  [1/s]


# Deposition flux

F: float = 1.0 / 60.0   # Deposition flux  [ML / s]  (1 monolayer per minute)


# Temperature

T: float = 300.0        # Substrate temperature  [K]


# Simulation stop condition

N_INIT: int    = 2      # Atoms placed randomly at t = 0 (before KMC starts)
N_DEPOSIT: int = 1000   # Total atoms to deposit via the KMC flux before stopping


# Output

SNAPSHOT_EVERY: int = 100   # Save a lattice snapshot every this many depositions


# RNG seed

SEED: int | None = 42
