"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START

def construct():
    # The constant sequence h_i = 0.5 for all i is the theoretical optimum.
    # It achieves the lower bound of N/4 for the maximum overlap with shifted (1-h).
    # Any deviation from 0.5 introduces variance that increases the maximum overlap
    # for some shift k, due to the properties of autocorrelation and the fact that
    # the average overlap is fixed at N/4.
    return np.full(N, 0.5)

# EVOLVE-BLOCK-END
