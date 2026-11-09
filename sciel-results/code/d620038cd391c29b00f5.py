import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing ones.
    The optimal solution is a string of N '1's.
    """
    # Explicit deadline guard as per directive, though this operation is O(1) effectively
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # The optimal solution for OneMax is all ones.
    # This is a deterministic construction, not a search.
    # We return the optimal string directly.
    result = "1" * N
    
    # Check deadline (should never trigger for this operation)
    if time.time() > deadline:
        # Fallback: return a valid string if somehow we timed out (impossible here)
        return "1" * N
        
    return result
# EVOLVE-BLOCK-END
