"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""

import time

N = 100
DEADLINE_SECONDS = 120.0

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the optimal 100-bit binary string for OneMax.
    Returns the global optimum directly to minimize overhead.
    """
    start_time = time.time()
    
    # The global optimum for OneMax is a string of all '1's.
    # Direct construction avoids any loop or join overhead.
    result = '1' * N
    
    # Explicit deadline guard: check if we are close to the limit.
    # For this direct construction, we are always well within limits,
    # but this structure ensures compliance with the compute budget directive.
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 1.0:
        # If we were doing iterative search, we would return the best found so far.
        # Here, the result is already the global optimum.
        pass
        
    return result
# EVOLVE-BLOCK-END
