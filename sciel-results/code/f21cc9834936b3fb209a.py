import numpy as np

N = 100

# Pre-allocate a NumPy array of ones at module load time
# Using int8 is sufficient and memory efficient
_ones_array = np.ones(N, dtype=np.int8)

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Returns a string of '1's of length N.
    """
    # Convert the pre-allocated array of ones to a binary string.
    # map(str, _ones_array) iterates over the numpy array, converting each element to string.
    # ''.join concatenates them.
    return ''.join(map(str, _ones_array))
# EVOLVE-BLOCK-END
