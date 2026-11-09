"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100
DEADLINE_MARGIN = 5.0  # seconds
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    # Construct the global optimum: a 100-bit binary string of all ones
    # This maximizes the number of ones (the LMX OneMax objective)
    
    # Explicit deadline guard
    # Check if we are close to the timeout (150s limit, margin 5s)
    if time.time() - START_TIME > 145.0:
        # Return best result found (which is the optimal string)
        # Fallback to standard string multiplication if numpy fails or for safety
        return "1" * N
        
    try:
        # Strategy: Represent the 100-bit vector as a NumPy array of uint8 initialized to 1
        # Then cast it to a string using np.char.mod
        # This leverages vectorized memory allocation and C-level string formatting
        arr = np.ones(N, dtype=np.uint8)
        # np.char.mod('1', arr) creates an array of strings '1'
        # Joining them gives the final binary string
        return ''.join(np.char.mod('1', arr))
    except Exception:
        # Fallback to standard Python string repetition if NumPy approach fails
        return "1" * N
# EVOLVE-BLOCK-END
