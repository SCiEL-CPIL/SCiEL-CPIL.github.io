"""OneMax 100-bit direct binary construction: return all ones."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution for OneMax is all '1's.
    Returns a string of length N consisting entirely of '1' characters.
    """
    return "1" * N
# EVOLVE-BLOCK-END
