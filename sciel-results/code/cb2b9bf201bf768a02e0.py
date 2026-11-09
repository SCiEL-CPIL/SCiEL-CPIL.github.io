import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The objective is OneMax: maximize the count of '1' characters in a
    binary string of length N. The optimal solution is all ones.
    
    This function returns the analytical optimum directly.
    A deadline guard and length verification are included for robustness.
    """
    # Deadline guard: The task limit is 120s. We set a generous margin.
    # The operation is O(N) with N=100, so it completes in microseconds.
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin, well within limits
    
    # Analytical solution for OneMax: all bits set to 1
    result = "1" * N
    
    # Defensive check: Ensure the length is exactly N
    if len(result) != N:
        # This should never happen for "1" * N, but ensures validity
        result = result[:N]
        
    # Check deadline (should never trigger for this trivial case)
    if time.time() > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END
