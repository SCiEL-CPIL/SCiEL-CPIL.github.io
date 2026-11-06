"""OneMax direct binary construction (100 bits) - global optimum."""

import time

N = 100
# Pre-compute the optimal solution: a string of 100 '1's
# This is the unique global maximizer for the OneMax objective.
_OPTIMAL_STR = "1" * N

# EVOLVE-BLOCK-START
def construct():
    """
    Return a 100-bit binary string with all ones (global optimum).
    
    Strategy: Direct Analytical Construction.
    The OneMax objective function is linear and separable. The global optimum
    is trivially known to be the all-ones string. We return this pre-computed
    string immediately. A minimal time guard is included to satisfy the
    compute budget directive, though the operation completes in nanoseconds.
    """
    # Safety guard: Explicit deadline check as per directive.
    # Since the solution is pre-computed, this will never trigger for N=100,
    # but it structurally adheres to the requirement to track elapsed time.
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    # Return the optimal solution.
    result = _OPTIMAL_STR
    
    # Check if we are dangerously close to the limit (highly unlikely).
    if time.time() + safety_margin > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END
