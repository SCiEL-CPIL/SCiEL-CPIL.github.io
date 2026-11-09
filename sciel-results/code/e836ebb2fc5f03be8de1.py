import time

# EVOLVE-BLOCK-START
def construct():
    # The optimal set of n distinct positive integers with distinct subset sums
    # is the set of powers of 2: {1, 2, 4, ..., 2^(n-1)}.
    # For n=12, the largest element is 2^11 = 2048.
    # This is the theoretical minimum.
    
    n = 12
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
