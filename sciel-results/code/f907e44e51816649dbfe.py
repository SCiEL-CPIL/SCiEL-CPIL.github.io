"""Optimized OneMax constructor: direct binary construction maximizing ones."""

import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The OneMax objective is to maximize the count of '1' bits.
    The global optimum is a string of length N consisting entirely of '1's.
    This function directly constructs this optimal string using a NumPy array
    of uint8 values initialized to 1, then casts to a string.
    
    Returns:
        str: A binary string of length 100 consisting entirely of '1's.
    """
    # Deadline guard for safety
    start_time = time.time()
    deadline = start_time + 119.0  # 1 second safety margin before 120s limit
    
    # The optimal solution for OneMax is all ones.
    # Use NumPy vectorized allocation and string formatting.
    arr = np.ones(N, dtype=np.uint8)
    # Convert to string of '1's
    result = ''.join(np.char.mod('1', arr).astype(str))
    
    # Check deadline (though this operation is instantaneous)
    if time.time() > deadline:
        # Fallback to simple string multiplication if NumPy path was slow
        return "1" * N
        
    return result
# EVOLVE-BLOCK-END
