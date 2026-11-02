import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Generate a random sequence of -1 and 1
    rng = np.random.default_rng(42)
    heights = rng.integers(0, 2, N) * 2 - 1
    return heights
# EVOLVE-BLOCK-END
