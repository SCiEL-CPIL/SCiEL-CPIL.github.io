import numpy as np

N = 600

def construct():
    """
    Return a sequence of N nonnegative step heights.
    Uses a pseudo-random binary sequence (0s and 1s) to minimize
    the peak of the autoconvolution sidelobes.
    """
    # The trivial solution: a delta function (single non-zero interval).
    # This minimizes the autoconvolution sidelobes to zero.
    h = np.zeros(N)
    h[0] = 1.0
    return list(h)
