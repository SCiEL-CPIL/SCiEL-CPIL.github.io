"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The OneMax objective is simply the count of '1' characters in the string.
    The global maximum is achieved by a string of all '1's.
    We return this optimal solution directly.
    """
    # Explicit deadline guard (safety margin for 150s limit, though this is instantaneous)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # The optimal solution for OneMax is a string of all ones
    # This is O(N) to construct and O(1) to determine
    optimal_string = "1" * N
    
    # Check deadline (will always pass for N=100)
    if time.time() > deadline:
        # In case of unexpected slowness (impossible here), return what we have
        return optimal_string
        
    return optimal_string
# EVOLVE-BLOCK-END
