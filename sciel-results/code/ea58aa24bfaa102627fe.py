"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard to ensure we finish well within the 150s limit
    t0 = time.perf_counter()
    limit = t0 + 145.0
    
    # Strategy: Model the binary string as a NumPy boolean array initialized to True.
    # This leverages vectorized memory representation and a distinct computational path
    # from string multiplication.
    
    # Create a NumPy array of booleans, all True
    bits = np.ones(N, dtype=bool)
    
    # Convert the boolean array to a string of '1's
    # np.char.mod or similar can be used, but a simple join or list comprehension 
    # over the array is straightforward. However, to leverage NumPy, we can map
    # True->'1' and False->'0'. Since all are True, we just need '1'*N, but let's
    # use a NumPy-native conversion if possible, or a fast loop.
    # Actually, the most direct NumPy-to-string for a 1D bool array representing bits:
    # We can use np.where or a simple list comp. For speed, since we know they are all True,
    # we could just return '1'*N, but the directive asks for the NumPy path.
    # Let's use a NumPy-aware string construction:
    # np.array2string is not ideal for this.
    # A common trick: np.array(bits, dtype=str) doesn't work directly to '1'/'0'.
    # Let's use a list comprehension which is fast enough for N=100, or 
    # we can map the bool array to a char array.
    
    # Efficient conversion: map True to '1' and False to '0'
    # For N=100, a simple list comp is very fast.
    optimal_str = ''.join('1' if b else '0' for b in bits)
    
    # Safety check
    if time.perf_counter() > limit:
        return optimal_str
        
    return optimal_str
# EVOLVE-BLOCK-END
