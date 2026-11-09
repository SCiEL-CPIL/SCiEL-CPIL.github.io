N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The optimal solution for OneMax is the all-ones string.
    This is deterministic and runs in O(1) time (constant string creation).
    """
    return "1" * N
# EVOLVE-BLOCK-END
