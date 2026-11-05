"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The OneMax objective is to maximize the number of 1s in a binary string.
    The global optimum is a string of all 1s, which yields a score of N.
    This is achieved in constant time with no constraints other than length.
    """
    # Track elapsed time for safety margin (though this operation is O(N) and instantaneous)
    start_time = time.time()
    deadline = start_time + 115  # 5-second safety margin before 120s limit
    
    # The optimal solution for OneMax is all ones
    result = "1" * N
    
    # Verify length (sanity check, should always be true)
    assert len(result) == N
    
    # Return immediately; no search needed
    return result
# EVOLVE-BLOCK-END
