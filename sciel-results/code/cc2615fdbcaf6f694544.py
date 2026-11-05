"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a binary string of length N with all bits set to 1.
    
    Strategy: Direct construction using string multiplication.
    This is the most efficient method to generate the global optimum
    (all ones) for the OneMax problem. It runs in O(N) time with a very
    low constant factor, ensuring it completes well within the time limit.
    
    The solution is analytically derived: to maximize the number of ones
    in a fixed-length binary string, every bit should be 1.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin of 5 seconds before 150s limit
    
    # Direct construction of the optimal solution
    # '1' * N creates a string of N ones
    result = "1" * N
    
    # Ensure the length is exactly N (sanity check, though "1" * N is always length N)
    if len(result) != N:
        # Fallback logic, though theoretically unreachable
        result = result[:N].ljust(N, '1') if len(result) > N else result.ljust(N, '1')
        
    # Explicit deadline guard as per directive
    # In practice, this returns immediately
    if time.time() > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END
