"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The objective is OneMax: maximize the number of '1's in a binary string of length N.
    The global optimum is a string of all '1's.
    """
    return "1" * N
# EVOLVE-BLOCK-END
