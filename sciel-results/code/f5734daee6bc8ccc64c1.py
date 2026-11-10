"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a list of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    
    We use the arithmetic progression {226, 227, ..., 256} of size 31.
    This set satisfies max(sum of k largest) < min(sum of k+1 smallest)
    for all k, ensuring the condition.
    """
    return list(range(226, 257))

# EVOLVE-BLOCK-END
