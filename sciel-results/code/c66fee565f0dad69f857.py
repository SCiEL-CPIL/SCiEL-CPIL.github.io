N = 100

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    The global optimum for the OneMax problem is a string of all '1's.
    This solution returns the optimal string directly, ensuring maximum score
    with minimal execution time.
    """
    # Explicit deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # The solution is O(1), so we can just return it.
    # But we check the deadline to satisfy the directive.
    if time.time() < deadline:
        return "1" * N
    else:
        # Fallback (should never happen)
        return "1" * N
# EVOLVE-BLOCK-END
