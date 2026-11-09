"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The OneMax objective is maximized by a string of all '1's.
    This function directly returns the global optimum.
    """
    # The global optimum for OneMax with N bits is N ones.
    return "1" * N
# EVOLVE-BLOCK-END
