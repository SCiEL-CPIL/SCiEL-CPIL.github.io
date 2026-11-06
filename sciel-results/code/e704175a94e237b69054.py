"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution is trivially all ones since the objective
    is to maximize the count of '1' bits in a fixed-length binary string.
    """
    return "1" * N
# EVOLVE-BLOCK-END
