"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100
DEADLINE = 150  # Seconds wall-clock limit
START_TIME = time.time()

# Precompute the optimal solution once at module load for instant fallback
_OPTIMAL = "1" * N

# EVOLVE-BLOCK-START
def construct():
    """
    Direct construction: OneMax on 100 bits.
    The optimal solution is a string of 100 '1's.
    Uses integer-mask construction as the primary path with a
    module-level deadline guard and precomputed fallback.
    """
    # Deadline guard: if we are too close to the wall-clock limit,
    # return the precomputed optimal string immediately.
    if time.time() - START_TIME > DEADLINE - 10:
        return _OPTIMAL

    # Primary construction path: integer-to-binary conversion.
    # (1 << N) - 1 has all N low bits set to 1.
    mask = (1 << N) - 1
    result = bin(mask)[2:].zfill(N)

    # Sanity check: ensure length is exactly N
    if len(result) != N:
        return _OPTIMAL

    return result
# EVOLVE-BLOCK-END
