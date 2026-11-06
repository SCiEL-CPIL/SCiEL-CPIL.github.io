"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of 12 distinct positive integers such that all subset sums are distinct.
    The sequence of powers of 2 is the unique optimal solution for minimizing the largest element.
    For n=12, the largest element is 2^11 = 2048.
    """
    n = 12
    # The minimal superincreasing sequence is 1, 2, 4, ..., 2^(n-1)
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
