import time
import numpy as np

N = 100
DEADLINE_SECONDS = 145.0

# EVOLVE-BLOCK-START
def construct():
    """
    OneMax: maximize the number of 1s in a binary string of length N.
    Direct construction: the optimal solution is all ones.
    """
    t0 = time.monotonic()
    
    # Optimal solution for OneMax is all ones
    # Using NumPy boolean array initialized to True (1s)
    bits = np.ones(N, dtype=bool)
    
    # Convert boolean array to string of '1's
    result = ''.join(['1' if b else '0' for b in bits])
    
    # Deadline guard: if we've exceeded our safety margin, return what we have
    if time.monotonic() - t0 > DEADLINE_SECONDS:
        return result
    
    return result
# EVOLVE-BLOCK-END
