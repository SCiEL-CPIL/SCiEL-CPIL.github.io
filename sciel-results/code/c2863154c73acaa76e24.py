"""
Optimized program for OneMax task using direct integer bit-manipulation.
Constructs the all-ones binary string via integer formatting.
"""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a binary string of length N with all bits set to '1'.
    Uses direct string construction for maximum efficiency.
    """
    return '1' * N
# EVOLVE-BLOCK-END
