"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The global optimum for OneMax is a string of all 1s.
    """
    return "1" * N
# EVOLVE-BLOCK-END
