import time

N = 100
# Pre-compute the optimal solution at import time.
# This is the global optimum for OneMax: all bits are 1.
_OPTIMAL_STR = "1" * N

# Deadline guard setup
_START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def construct():
    """
    Return a binary string of length N maximizing the number of ones.
    
    This implementation uses direct construction of the known global optimum.
    The result is pre-computed at module load time for O(1) retrieval.
    An explicit deadline guard is included to ensure compliance with wall-clock limits,
    although the operation is constant time.
    """
    # Check deadline guard
    # If we are close to the limit, we still return the optimal solution because it is ready.
    # This guard is primarily to satisfy the structural requirement for a deadline check.
    if time.time() - _START_TIME > (TIME_LIMIT - SAFETY_MARGIN):
        # In a search context, we would return best-so-far.
        # Here, we return the pre-computed optimum.
        return _OPTIMAL_STR
    
    return _OPTIMAL_STR
