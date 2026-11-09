"""Program for OneMax direct binary construction (100 bits)."""
import time

N = 100
_START = time.monotonic()
_DEADLINE = 115.0  # safety margin before the 120s wall-clock kill

# EVOLVE-BLOCK-START
def construct():
    """
    Return the unique global optimum for the 100-bit OneMax genotype task:
    a binary string of length N consisting entirely of '1' characters.

    The OneMax objective is sum(x_i) for x_i in {0,1}, i=1..N, with the
    unique maximizer at x_i = 1 for all i. Direct construction is optimal
    and incurs no search overhead. An explicit deadline guard satisfies
    the compute-budget contract without affecting the result.
    """
    # Direct construction of the known global optimum.
    best = '1' * N

    # Explicit deadline guard: if elapsed wall-clock exceeds the safety
    # margin, return immediately to guarantee a valid result before the
    # evaluator kills the process. In practice this branch is never taken
    # because string construction is O(N) with negligible cost.
    if (time.monotonic() - _START) > _DEADLINE:
        return best

    return best
# EVOLVE-BLOCK-END
