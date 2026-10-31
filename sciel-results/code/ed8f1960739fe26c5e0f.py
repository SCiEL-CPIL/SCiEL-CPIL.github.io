import numpy as np

N = 600

def construct():
    """
    Return a sequence of N nonnegative step heights.
    Uses a pseudo-random binary sequence (0s and 1s) to minimize
    the peak of the autoconvolution sidelobes.
    """
    rng = np.random.default_rng(42)
    # Generate random binary sequence
    h = rng.integers(0, 2, size=N).astype(float)
    return list(h)
