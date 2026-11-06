"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The objective is OneMax: maximize the count of '1' bits in a binary string.
    The optimal solution is trivially a string of all '1's.
    
    We return the optimal solution directly.
    """
    return "1" * N
# EVOLVE-BLOCK-END
