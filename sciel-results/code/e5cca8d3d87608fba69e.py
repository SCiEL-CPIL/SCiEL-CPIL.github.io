"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Theoretical analysis shows that for any real-valued sequence f of length N,
    # max|autocorr(f)| >= sum(f^2) >= (sum(f))^2 / N.
    # The constant sequence achieves this lower bound exactly.
    # Thus, the constant vector is the global minimizer.
    return np.ones(N, dtype=np.float64)
# EVOLVE-BLOCK-END
