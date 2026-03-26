"""
Main driver for reticular gas simulation
"""

import numpy as np

from initialization import empty_grating
from visualization import visualizza_reticolo

def main() -> None:

    grating = empty_grating
    visualizza_reticolo(grating)

    
if __name__ == "__main__":
    main()


