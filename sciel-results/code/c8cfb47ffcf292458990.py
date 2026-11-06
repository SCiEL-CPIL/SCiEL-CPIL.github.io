"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns the set of 12 distinct positive integers with distinct subset sums
    that minimizes the largest element.
    
    The optimal sequence is powers of 2: 1, 2, 4, ..., 2^11.
    The largest element is 2048.
    """
    n = 12
    result = []
    current_val = 1
    for i in range(n):
        result.append(current_val)
        current_val <<= 1
    return result
# EVOLVE-BLOCK-END
