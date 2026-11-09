N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    Uses the analytical solution: all bits set to 1.
    """
    # Analytical optimum for OneMax: all ones
    # This operation is O(N) and completes in microseconds
    return "1" * N
# EVOLVE-BLOCK-END
