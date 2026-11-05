import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    The optimal solution is all ones.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # The global optimum for OneMax is all bits set to 1.
    # We construct this directly.
    optimal_string = "1" * N
    
    # Check deadline (though this returns instantly)
    if time.time() > deadline:
        # Fallback to a valid string if somehow we timed out (impossible here)
        return "1" * N
        
    return optimal_string
# EVOLVE-BLOCK-END
