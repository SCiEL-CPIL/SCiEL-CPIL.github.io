"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Directly constructs the optimal 100-bit binary string for OneMax.
    Uses bitwise OR with a mask to generate the all-ones state.
    """
    # Mask for N bits set to 1
    mask = (1 << N) - 1
    # Initialize to 0 and OR with mask to get all ones
    val = 0
    val |= mask
    # Convert to binary string, stripping '0b' prefix
    return bin(val)[2:]
# EVOLVE-BLOCK-END
