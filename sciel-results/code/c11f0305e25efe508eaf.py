"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # The OneMax objective is simply the count of '1's in the binary string.
    # The maximum possible score for a string of length N is N, achieved by all '1's.
    # Since this is a direct construction task (not an evolutionary search),
    # we can directly return the optimal solution.
    
    # Deadline guard (safety margin before 150s limit)
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Directly construct the optimal solution: all ones
    result = "1" * N
    
    # Safety check (will never trigger for O(N) construction)
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
