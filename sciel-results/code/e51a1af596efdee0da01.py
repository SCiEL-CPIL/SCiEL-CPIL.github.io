import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate Thue-Morse sequence
    # t_n = popcount(n) % 2
    # We want values in {-1, 1}
    # Map 0 -> 1, 1 -> -1
    
    # Efficient generation for N=400
    # Can use bit_count in Python 3.10+ or numpy tricks
    # Numpy: use bit operations on arange
    
    indices = np.arange(N)
    
    # Calculate popcount for each index
    # Method: iterative bit counting or using a lookup table
    # Since N is small (400), simple loop or vectorized bit ops are fine.
    
    # Vectorized popcount approximation for small N
    # Or just use a list comprehension with bin().count('1')
    # 400 is tiny.
    
    tm_bits = [bin(i).count('1') % 2 for i in range(N)]
    
    # Map to {-1, 1}
    # 0 -> 1, 1 -> -1
    seq = np.array([1 if b == 0 else -1 for b in tm_bits], dtype=np.float64)
    
    # Check time
    if time.time() > deadline:
        # Return best found (which is the only one)
        return seq
        
    return seq
# EVOLVE-BLOCK-END
