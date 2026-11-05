"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a binary string of length N with all bits set to 1.
    This is the global optimum for the OneMax objective.
    """
    # The optimal solution for OneMax is a string of all ones.
    # This takes O(N) time to construct, which is negligible.
    return "1" * N
# EVOLVE-BLOCK-END
