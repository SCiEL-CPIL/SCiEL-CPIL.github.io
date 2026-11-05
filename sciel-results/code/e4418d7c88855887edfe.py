"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    This function constructs the global optimum for the OneMax problem, which is
    a string of N '1's. It uses direct string construction for maximum efficiency
    and includes an explicit deadline guard to ensure compliance with time limits,
    although the computation is constant time.
    """
    start_time = time.time()
    # Set a safety margin before the 150s limit. 
    # The operation is O(N) with N=100, so this is just a formality.
    deadline = start_time + 145.0 

    # The optimal solution for OneMax is a string of all ones.
    # Direct string construction is the most efficient method.
    result = "1" * N
    
    # Explicit deadline guard
    if time.time() > deadline:
        # If we somehow exceed the deadline (extremely unlikely), return the 
        # valid all-ones string directly to avoid timeout.
        return result
        
    return result
# EVOLVE-BLOCK-END
