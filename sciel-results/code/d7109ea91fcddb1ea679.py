"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100
DEADLINE = 150  # Seconds
START_TIME = time.time()

# Precompute the optimal string once at module load
_OPTIMAL = "1" * N

# EVOLVE-BLOCK-START
def construct():
    """
    Direct construction: OneMax on 100 bits.
    Returns the precomputed optimal string of 100 ones.
    Deadline guard returns the same optimal result if time is nearly exhausted.
    """
    # Explicit deadline guard: return best known (optimal) result with safety margin
    if time.time() - START_TIME > DEADLINE - 10:
        return _OPTIMAL
    # Construct optimal solution (precomputed for minimal overhead)
    return _OPTIMAL
# EVOLVE-BLOCK-END
