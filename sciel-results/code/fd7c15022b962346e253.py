"""OneMax direct binary construction (100 bits).

Constructs a 100-bit binary string maximizing ones.
The global optimum is all ones, giving a score of 100.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Returns a string of N '1' characters, which is the unique global optimum
    for the OneMax objective with N bits.
    
    Uses direct string repetition for maximum efficiency.
    """
    t0 = time.monotonic()
    # Safety margin: return well before the 150s wall-clock kill.
    deadline = t0 + 140.0

    # Direct construction of the optimal string
    optimal = "1" * N

    # Deadline guard: return the result if within time budget.
    if time.monotonic() < deadline:
        return optimal

    # Fallback (unreachable for O(1) work, but satisfies contract):
    return optimal
# EVOLVE-BLOCK-END
