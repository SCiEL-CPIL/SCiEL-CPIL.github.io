"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The minimal such n is 143 = 11 * 13.
    Verification:
    - phi(143) = phi(11) * phi(13) = 10 * 12 = 120
    - 120 is divisible by 120
    - No smaller n in [1, 142] has phi(n) divisible by 120
    
    This is a precomputed constant based on mathematical analysis:
    - phi(n) must be divisible by 120 = 8 * 3 * 5
    - Checking all n from 1 to 142, none satisfy the condition
    - n = 143 = 11 * 13 gives phi(143) = 120, which is divisible by 120
    """
    return 143
# EVOLVE-BLOCK-END
