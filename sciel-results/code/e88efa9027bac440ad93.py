"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# Pre-allocate a NumPy array of ones at module load time
_ones_array = np.ones(N, dtype=np.int8)

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Uses a pre-allocated NumPy array of ones, converting to string only upon invocation.
    """
    # Convert the pre-allocated array of ones to a binary string
    return ''.join(str(int(x)) for x in _ones_array)
# EVOLVE-BLOCK-END
