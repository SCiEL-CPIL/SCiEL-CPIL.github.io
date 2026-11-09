"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the optimal 100-bit binary string for the OneMax problem.
    The OneMax objective is to maximize the number of ones in the string.
    The global optimum is a string of all ones.
    """
    # Explicit deadline guard as per directive, though computation is O(1).
    # This ensures we return within the 120s limit (or 150s kill switch).
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # The optimal solution for OneMax is all 1s.
    # This is the global maximum.
    optimal_string = "1" * N
    
    # Explicit deadline guard check
    # Since the operation is O(1), we are always within the deadline.
    # We return the optimal string regardless, satisfying the guard requirement
    # by having tracked the time.
    _ = time.time()  # Ensure time is checked/updated as per directive spirit
    return optimal_string
# EVOLVE-BLOCK-END
