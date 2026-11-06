"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Prioritizes vectors with lower Hamming weight.
    Vectors with fewer non-zero elements are considered 'safer' to add early
    in the greedy construction of a cap set, as they are less likely to
    form arithmetic progressions with other low-weight vectors immediately.
    This heuristic typically yields larger cap sets than uniform or lexicographic
    ordering.
    """
    # Calculate Hamming weight (number of non-zero elements)
    weight = 0
    for v in vector:
        if v != 0:
            weight += 1
    # Lower weight should have higher priority (larger value)
    # Using negative weight ensures that weight 0 > weight 1 > weight 2 ...
    return -float(weight)
# EVOLVE-BLOCK-END
