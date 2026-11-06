import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    The global optimum for the OneMax problem is a string of all '1's.
    
    This implementation uses a NumPy boolean array initialized to True 
    to leverage vectorized memory representation, then converts it to 
    a string of '1's. This fulfills the strategy to use array-based 
    construction while maintaining the optimal solution.
    """
    # Explicit deadline guard (safety margin)
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Check if we have time to proceed (always true for this O(N) operation)
    if time.time() < deadline:
        # Create a NumPy boolean array of length N, all True
        # This represents a binary string of all 1s
        arr = np.ones(N, dtype=bool)
        
        # Convert the boolean array to a string of '1's
        # np.char.add or similar vectorized ops can be used, but simple join is robust
        # Since arr is all True, we can just map True->'1', False->'0'
        # For efficiency with all-True, "1"*N is fastest, but we want to demonstrate the array path.
        # A direct conversion:
        result = ''.join('1' if x else '0' for x in arr)
        
        return result
    else:
        # Fallback (should never happen)
        return "1" * N
# EVOLVE-BLOCK-END
