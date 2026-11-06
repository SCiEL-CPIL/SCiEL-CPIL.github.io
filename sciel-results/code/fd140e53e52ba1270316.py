# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Uses a pre-allocated NumPy array of ones, converting to string only upon invocation.
    """
    # Convert the pre-allocated array of ones to a binary string using vectorized operations
    return ''.join(_ones_array.astype(str))
# EVOLVE-BLOCK-END
