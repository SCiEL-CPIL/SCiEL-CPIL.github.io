"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Analysis:
    We require 120 | phi(n).
    For n < 143, phi(n) < n < 143. The only multiple of 120 in (0, 143) is 120.
    So we need phi(n) = 120 for n < 143.
    Checking all n < 143, none satisfy phi(n) = 120.
    For n = 143 (11 * 13), phi(143) = 10 * 12 = 120.
    Thus, 143 is the minimum.
    """
    return 143
# EVOLVE-BLOCK-END
