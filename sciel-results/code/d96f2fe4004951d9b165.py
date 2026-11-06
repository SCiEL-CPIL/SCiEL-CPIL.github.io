"""Direct OneMax construction: return the optimal 100-bit string (all ones)."""
N = 100

# Pre-compute the optimal string at module load time to minimize per-call overhead
_OPTIMAL = "1" * N

# EVOLVE-BLOCK-START
def construct():
    """Return a 100-bit binary string maximizing the number of ones."""
    return _OPTIMAL
# EVOLVE-BLOCK-END
