"""
Constructs a 100-bit binary string maximizing the number of ones.
The optimal solution is a string of all ones.
"""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a binary string of length N consisting entirely of '1's.
    This maximizes the count of ones, which is the objective function.
    """
    return "1" * N
# EVOLVE-BLOCK-END
