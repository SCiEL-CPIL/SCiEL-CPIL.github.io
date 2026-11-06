"""
Direct construction for OneMax (100-bit).
The optimal solution is a string of 100 '1' characters.
We build it explicitly and return it immediately.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a binary string of length N with all bits set to '1'.
    This is the direct, optimal solution for the OneMax problem.
    """
    t0 = time.monotonic()
    wall_limit = 150.0
    margin = 5.0

    # Direct construction: all ones is the unique global optimum for OneMax.
    # This is the optimal solution.
    result = "1" * N

    # Explicit deadline guard
    elapsed = time.monotonic() - t0
    if elapsed + margin < wall_limit:
        return result
    else:
        # In the extremely unlikely event of timeout, return the best found (which is the same)
        return result
# EVOLVE-BLOCK-END
