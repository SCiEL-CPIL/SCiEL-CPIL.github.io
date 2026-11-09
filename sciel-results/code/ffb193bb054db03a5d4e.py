"""OneMax 100-bit binary construction: maximize number of ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The optimal solution is all ones. We return it directly.
    """
    # Explicit deadline guard (safety margin before 120s limit)
    start_time = time.time()
    deadline = 119.0
    
    # The optimal solution for OneMax is a string of all '1's.
    # This is an O(N) construction which is effectively O(1) for N=100.
    result = "1" * N
    
    # Check if we are still within the time limit (always true for this trivial case)
    if time.time() - start_time > deadline:
        # Fallback to a valid but suboptimal string if somehow we timed out
        # (This will never happen, but satisfies the guard requirement)
        return "1" * 50 + "0" * 50
        
    return result
# EVOLVE-BLOCK-END
