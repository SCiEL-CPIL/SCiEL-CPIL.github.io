"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    
    The set is an arithmetic progression {226, 227, ..., 256} of size 31.
    This satisfies the condition that the minimum sum of k elements is strictly
    greater than the maximum sum of k-1 elements for all k, ensuring no
    collisions between sums of different cardinalities.
    """
    # The set {226, 227, ..., 256} has 31 elements.
    # Min sum of k elements: sum_{i=0}^{k-1} (226+i)
    # Max sum of k-1 elements: sum_{i=0}^{k-2} (256-i)
    # Gap is positive for all k=1..31.
    return list(range(226, 257))

# EVOLVE-BLOCK-END
