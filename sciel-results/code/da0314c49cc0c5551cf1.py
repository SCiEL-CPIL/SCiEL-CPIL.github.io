"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns the set of denominators for the Egyptian fraction expansion of 5/31
    that minimizes the largest denominator.
    
    Optimal solution: 1/8 + 1/31 + 1/248 = 5/31
    Largest denominator: 248
    """
    # The denominators 8, 31, 248 sum to 5/31
    # 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31
    denominators = (8, 31, 248)
    return list(denominators)
# EVOLVE-BLOCK-END
