"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns the maximum set of distinct integers in [2, 40] such that all 
    subset products are distinct.
    
    The maximum size is 12, achieved by selecting the 12 primes in the range.
    Any set of 13 or more integers would contain a multiplicative dependency,
    leading to duplicate subset products.
    """
    # The 12 primes in the range [2, 40]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    return primes

# EVOLVE-BLOCK-END
