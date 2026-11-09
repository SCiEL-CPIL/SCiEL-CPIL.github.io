"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100
DEADLINE = 150  # Seconds
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    """
    Direct construction: OneMax on 100 bits.
    The optimal solution is a string of 100 '1's.
    Includes a deadline guard as per directive.
    Uses NumPy boolean array for construction.
    """
    # Check time elapsed
    if time.time() - START_TIME > DEADLINE - 10:
        # Return best known result (which is optimal)
        return "1" * N
    
    # Construct optimal solution using NumPy boolean array
    # Initialize boolean array with True (representing 1)
    bool_arr = np.ones(N, dtype=bool)
    # Convert boolean array to string array of '1's and '0's
    # np.where(bool_arr, '1', '0') creates an array of characters
    char_arr = np.where(bool_arr, '1', '0')
    # Join the character array into a single string
    return "".join(char_arr)
# EVOLVE-BLOCK-END
