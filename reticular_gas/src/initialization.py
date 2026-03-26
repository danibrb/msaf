""""
Generating starting position of particles, using random generator
"""


import numpy as np

from config import L

def empty_grating(dim: int) -> np.ndarray:
    shape = (dim, dim)
    return np.zeros((shape), dtype= bool)

def generate_all_labels(dim: int) -> np.ndarray:
    return np.arange(L * L)

def shuffle_labels(labels: np.ndarray) -> np.ndarray:
    return np.random.shuffle(labels)

lab = generate_all_labels(L)
s_lab = shuffle_labels(lab)
print(lab)
print("shuffled")
print(s_lab)