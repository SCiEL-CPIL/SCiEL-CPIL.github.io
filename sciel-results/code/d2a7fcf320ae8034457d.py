import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a binary string of length N with all bits set to '1'.
    This is the direct, optimal solution for the OneMax problem.
    """
    start_time = time.time()
    deadline = 150.0  # Wall-clock limit
    safety_margin = 5.0

    # Direct construction: all ones is the unique global optimum for OneMax.
    result = "1" * N

    # Safety guard: ensure we return before the wall-clock deadline.
    if time.time() + safety_margin > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
