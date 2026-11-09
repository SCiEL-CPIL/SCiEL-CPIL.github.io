"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a set of 12 distinct positive integers such that all 4096 subset sums are distinct.
    The set minimizes the largest integer.
    
    The optimal set is the powers of 2: {1, 2, 4, ..., 2048}.
    Largest integer is 2048.
    """
    return [1 << i for i in range(12)]
# EVOLVE-BLOCK-END
