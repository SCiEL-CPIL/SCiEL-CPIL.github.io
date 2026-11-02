"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Use a symmetric unimodal step function (triangular) to minimize
    # the normalized max autoconvolution. For nonnegative functions,
    # the autoconvolution is maximized at 0, giving ratio 1.
    # A triangular profile ensures this property holds tightly.
    t = np.linspace(-1, 1, N)
    heights = np.maximum(0, 1 - np.abs(t))
    # Ensure nonnegative and avoid exact zeros causing issues
    heights = np.maximum(heights, 1e-10)
    return heights.tolist()
# EVOLVE-BLOCK-END
