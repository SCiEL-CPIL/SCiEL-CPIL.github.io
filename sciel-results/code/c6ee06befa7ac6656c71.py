"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120."""
    # The smallest n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11 * 13) = (11-1) * (13-1) = 10 * 12 = 120.
    # All n < 143 have phi(n) not divisible by 120.
    return 143
# EVOLVE-BLOCK-END
