"""
Optimal OneMax solution: Construct the string of all 1s.

This program directly constructs the global optimum for the OneMax problem
with N=100 bits. The objective is to maximize the number of ones, which is
achieved by setting every bit to 1.

The implementation includes an explicit deadline guard to ensure compliance
with the evaluator's wall-clock time limit (150s kill, 120s soft limit implied).
Since the computation is trivial (O(N) string construction), the guard is a
safety net to prevent any unexpected overhead from causing a timeout.
"""

import time

N = 100

def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The global optimum for OneMax is a string consisting entirely of '1's.
    This function directly returns that string.
    
    An explicit deadline guard is implemented to track elapsed time and
    return the best result (which is the optimal result) with a safety
    margin before the limit.
    """
    # Explicit deadline guard: Track time from the start of this function call.
    # We use a safety margin of 10 seconds below the 120s soft limit (or 150s hard kill).
    # 110s is a conservative margin that allows for any import/setup delays not captured here.
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Direct construction of the optimal solution: all bits set to 1.
    # This is the global optimum for OneMax with N=100.
    result = "1" * N
    
    # Safety check: ensure we are within the deadline.
    # For this trivial task, this check will always pass.
    if time.time() < deadline:
        return result
    else:
        # Fallback: return the optimal result even if time limit is exceeded,
        # as the computation is already complete.
        return result
