import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    The optimal solution is all ones.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # The global optimum for OneMax is all bits set to 1.
    # We construct this directly.
    # This operation is O(N) and extremely fast, but we respect the budget structure.
    result = "1" * N
    
    # Check deadline (will never trigger for this trivial op, but ensures compliance)
    if time.time() > deadline:
        # Fallback: return best found (which is the same)
        pass
        
    return result
# EVOLVE-BLOCK-END
