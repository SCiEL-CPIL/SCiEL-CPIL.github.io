"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The objective is to maximize the number of ones in a 100-bit binary string.
    The global optimum is trivially the string of all ones, which has 100 ones.
    
    Since this is a direct construction task (not an evolutionary search),
    we can simply return the known optimal solution immediately.
    This eliminates any stochastic variance and runtime overhead.
    
    A deadline guard is included for safety, though the computation is O(1).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin: 140s out of 150s limit
    
    # The optimal solution for OneMax is all ones
    result = "1" * N
    
    # Verify we're within time budget (should always be true for O(1) work)
    if time.time() > deadline:
        # Fallback: return a valid string (still all ones, but this path is essentially unreachable)
        return "1" * N
    
    return result
# EVOLVE-BLOCK-END
