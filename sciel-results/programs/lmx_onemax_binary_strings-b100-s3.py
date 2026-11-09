"""Program for OneMax direct binary construction (100 bits)."""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The task is OneMax: maximize the number of '1' bits in a binary string of length N.
    The direct, optimal construction is to set all bits to '1'.
    This achieves the maximum possible score of N (100 ones).
    """
    # Direct approach: all ones is the optimal solution for OneMax
    return "1" * N
# EVOLVE-BLOCK-END
