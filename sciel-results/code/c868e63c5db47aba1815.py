"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The objective is to maximize the number of ones in a 100-bit string.
    The global optimum is the string of all ones, which has exactly 100 ones.
    
    Direct construction of this known optimum:
    - Eliminates stochastic variance
    - Zero runtime overhead (O(1) string creation)
    - Guarantees the maximum possible score
    
    Deadline guard: track elapsed time and return best result with safety margin.
    Since we return the exact optimum immediately, no search loop is needed.
    """
    start = time.time()
    deadline = start + 145.0  # 5s safety margin before 150s kill
    
    # The optimal solution is all ones
    result = "1" * N
    
    # Verify we haven't exceeded the deadline (should never happen for O(1) work)
    if time.time() > deadline:
        # This branch is unreachable for the direct construction,
        # but included as a safety guard per the compute budget directive
        return result
    
    return result
# EVOLVE-BLOCK-END
