"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Analysis:
    We require phi(n) % 120 == 0.
    The smallest positive multiple of 120 is 120.
    We search for the smallest n such that phi(n) = 120.
    Candidates for phi(n) = 120:
    n=143 (11*13) -> phi(143) = 10*12 = 120.
    Checking n < 143:
    If phi(n) = 120, n must be > 120.
    We verify no n in [1, 142] has phi(n) as a multiple of 120.
    Since phi(n) < n for n > 1, phi(n) cannot be 240, 360, etc. for n < 143.
    Thus, we only need to check if phi(n) = 120 for n < 143.
    It has been verified that 143 is the smallest such integer.
    
    Therefore, the minimum n is 143.
    """
    return 143

# EVOLVE-BLOCK-END
