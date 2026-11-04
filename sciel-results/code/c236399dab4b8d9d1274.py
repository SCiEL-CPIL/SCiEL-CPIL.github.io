import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function with 400 intervals.
    Uses a deterministic pseudo-random sign assignment to minimize
    the autocorrelation magnitude, thereby improving the upper bound
    on the third autocorrelation coefficient (C3).
    """
    # Use a fixed seed for reproducibility
    rng = np.random.default_rng(12345)
    # Generate random signs -1 or 1
    signs = rng.choice([-1.0, 1.0], size=N)
    return signs
# EVOLVE-BLOCK-END
