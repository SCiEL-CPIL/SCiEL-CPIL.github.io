import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    
    For the OneMax problem with a linear objective function where each '1' contributes
    positively to the score, the global optimum is a string consisting entirely of '1's.
    Therefore, we directly construct this string.
    
    A deadline guard is included to ensure compliance with the 120s/150s time limits,
    although the operation is O(N) and completes in microseconds.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin well below 120s/150s limits
    
    # The optimal solution for OneMax is a string of all 1s.
    # This is O(N) to construct and returns the maximum possible score.
    result = '1' * N
    
    # Explicit deadline check (though not strictly necessary for O(N) string creation)
    if time.time() > deadline:
        # Return the best result found so far (which is already the optimal)
        return result
    
    return result
# EVOLVE-BLOCK-END
