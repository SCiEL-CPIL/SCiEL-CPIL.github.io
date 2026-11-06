"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Since the objective is to maximize the number of ones,
    the optimal solution is all ones.
    """
    return "1" * N
# EVOLVE-BLOCK-END
