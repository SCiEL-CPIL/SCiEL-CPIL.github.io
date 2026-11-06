"""Direct OneMax construction: return all-ones string of length N."""
N = 100

# EVOLVE-BLOCK-START
# Pre-allocate the optimal string at module load time.
# The optimal OneMax solution is the all-ones string.
_OPTIMAL = "1" * N

def construct():
    """Return a 100-bit binary string maximizing the number of ones."""
    return _OPTIMAL
# EVOLVE-BLOCK-END
