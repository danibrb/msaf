""""
Generating starting position of particles, using random generator
"""


import numpy as np

from config import L

def empty_grating(l: int) -> np.ndarray:
    shape = (l, l)
    return np.zeros(shape)

print(empty_grating(int(L)))

