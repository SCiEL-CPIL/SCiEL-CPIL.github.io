"""OneMax 100-bit binary construction: maximize number of ones."""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The optimal solution is the string of all ones.
    Returns:
        str: A string of length N consisting entirely of '1's.
    """
    return '1' * N
# EVOLVE-BLOCK-END
