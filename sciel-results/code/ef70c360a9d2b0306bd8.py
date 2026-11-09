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
    
    Implementation strategy:
    1. Check deadline guard to ensure we return within time limits.
    2. Construct the optimal solution using a NumPy boolean array initialized to True.
    3. Convert the boolean array to a string of '1's using astype('U1') or similar vectorized method.
    4. Return the resulting string.
    """
    # Deadline guard
    if time.time() - START_TIME > DEADLINE - 10:
        # Fallback to direct string multiplication if time is critical
        return "1" * N
    
    # Construct optimal solution using NumPy boolean array
    # Initialize boolean array with True (representing 1)
    bool_arr = np.ones(N, dtype=bool)
    
    # Convert boolean array to a string of '1's
    # Using astype('U1') converts True to '1' and False to '0' in a vectorized manner
    # However, np.where is often more explicit and robust for character mapping
    # Let's use the suggested astype pathway or similar efficient conversion
    # np.array(['1'] * N) is also fast, but let's stick to the numerical pathway
    # A direct boolean to string conversion isn't a single numpy call that yields '1'/'0' directly without mapping.
    # np.where(bool_arr, '1', '0') is the standard vectorized way.
    char_arr = np.where(bool_arr, '1', '0')
    
    # Join the character array into a single string
    return "".join(char_arr)
# EVOLVE-BLOCK-END
