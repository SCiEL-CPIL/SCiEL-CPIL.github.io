"""
Optimal OneMax solution: Construct the string of all 1s.

Combines the direct construction strategy with a robust deadline guard.
The global optimum for OneMax is a string of all '1's, which is found immediately.
"""

import time

N = 100
START_TIME = time.time()
TIME_LIMIT = 150  # Seconds, matching the evaluator's kill limit

def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The global optimum for OneMax is a string consisting entirely of '1's.
    This function directly returns that string.
    
    A deadline guard is included to ensure the program returns a valid result
    even in the unlikely event that wall-clock time has exceeded the limit.
    Since the computation is O(N) and trivial, this guard is a safety net.
    """
    # Deadline guard: if we are already close to the time limit, return immediately.
    # This is a safety measure; for this task, the time check itself is negligible.
    if time.time() - START_TIME > TIME_LIMIT - 1:
        return "1" * N
        
    # Direct construction of the optimal solution: all bits set to 1.
    # This is the global optimum for OneMax with N=100.
    return "1" * N
