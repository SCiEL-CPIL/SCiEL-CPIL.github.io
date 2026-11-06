"""OneMax direct binary construction (100 bits) - global optimum."""
N = 100

# Pre-allocate the optimal string at module load time
_OPTIMAL = "1" * N

# EVOLVE-BLOCK-START
def construct():
    """Return a 100-bit binary string with all ones (global optimum).
    
    This solution achieves the theoretical maximum score of 100.
    No search or time-based heuristics are required as the global
    optimum is directly constructible.
    """
    return _OPTIMAL
# EVOLVE-BLOCK-END
