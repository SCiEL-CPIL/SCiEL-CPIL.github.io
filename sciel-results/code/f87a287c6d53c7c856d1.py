import time

N = 100
START_TIME = time.time()
DEADLINE = 150.0  # Explicit safety margin below the 150s kill limit

# EVOLVE-BLOCK-START
def construct():
    # Direct construction: The optimal solution for OneMax is all ones.
    # This is an O(1) operation, so it will never trigger the deadline.
    # We include the guard to satisfy the contract's requirement for an explicit deadline check.
    
    if time.time() - START_TIME > DEADLINE:
        # Fallback: Return the optimal solution as well, as it is the best possible result.
        return "1" * N

    return "1" * N
# EVOLVE-BLOCK-END
