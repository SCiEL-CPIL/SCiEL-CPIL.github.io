"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return integer n in [1,1000000] with phi(n) divisible by 120.
    Minimize n.
    
    The minimum n is 143, as phi(143) = phi(11*13) = 10*12 = 120.
    No smaller n has a totient divisible by 120.
    """
    return 143

# EVOLVE-BLOCK-END
