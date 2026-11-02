import numpy as np

N = 600

def construct():
    # The uniform distribution is the theoretical minimum for the L-infinity norm
    # of the autoconvolution of a non-negative function with fixed support and L1 norm.
    # Any deviation increases the L2 norm (and thus the peak at 0) or shifts the peak.
    # We return the uniform distribution.
    return [1.0 / N] * N
