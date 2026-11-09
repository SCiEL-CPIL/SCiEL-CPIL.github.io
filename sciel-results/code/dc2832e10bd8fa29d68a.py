"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

N = 100
DEADLINE = 150  # Seconds
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    """
    Direct construction: OneMax on 100 bits.
    The optimal solution is a string of 100 '1's.
    Includes a deadline guard as per directive.
    Uses a bit-packed construction path to produce the same optimal string
    with structurally distinct code (integer → binary string).
    """
    # Deadline guard: if we are too close to the wall-clock limit, return
    # the precomputed optimal string immediately.
    elapsed = time.time() - START_TIME
    if elapsed > DEADLINE - 10:
        return "1" * N

    # Construct the optimal 100-bit string via integer-to-binary conversion.
    # (2**100 - 1) has all 100 low bits set to 1.
    mask = (1 << N) - 1
    return bin(mask)[2:].zfill(N)

# EVOLVE-BLOCK-END
