"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The OneMax objective is to maximize the count of '1's in a binary string of length N.
    The global optimum is a string consisting entirely of '1's.
    This solution returns the global optimum directly using NumPy vectorization.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # The optimal solution is all ones.
    # Use NumPy to create the array and convert to string.
    # This leverages C-level operations for allocation and formatting.
    arr = np.ones(N, dtype=np.uint8)
    optimal_string = np.char.mod('1', arr).astype(str)
    
    # Explicit deadline guard
    if time.time() < deadline:
        return optimal_string
    else:
        # Fallback: return the best result found (which is optimal)
        return optimal_string
# EVOLVE-BLOCK-END
