"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Generate Thue-Morse sequence of length N
    # Thue-Morse sequence is defined by t_0 = 0, t_{2n} = t_n, t_{2n+1} = 1 - t_n
    # We map 0 -> 1 and 1 -> -1 to create a signed step function
    
    # Efficiently generate the first N bits of the Thue-Morse sequence
    # t_n is the number of 1s in the binary representation of n mod 2
    # We can compute this vectorially
    
    # Create array of indices
    indices = np.arange(N)
    
    # Calculate parity of the number of set bits (popcount) for each index
    # Using bit operations: x ^ (x >> 1) ^ (x >> 2) ...
    # A simple loop is fast enough for N=400, but vectorized is cleaner.
    # We can use the property that t_n = (n ^ (n >> 1) ^ (n >> 2) ^ ...).bit_count() % 2
    # However, numpy doesn't have bit_count directly.
    # Let's use a simple iterative generation or a lookup table for small N.
    
    # Iterative generation is trivial and fast for N=400
    tm = np.zeros(N, dtype=int)
    tm[0] = 0
    for i in range(1, N):
        # t_i = t_{i // 2} if i is even, 1 - t_{i // 2} if i is odd
        # Actually, the standard definition is:
        # t_0 = 0
        # t_{2n} = t_n
        # t_{2n+1} = 1 - t_n
        # This allows generating in order if we keep the previous values?
        # No, it's recursive on i//2.
        
        # Let's just use the popcount parity method which is independent.
        # For n < 400, we can just compute it in a loop.
        pass

    # Let's stick to the simplest deterministic high-contrast sequence:
    # Alternating +1, -1 is too simple (period 2).
    # Thue-Morse is the next level of complexity.
    
    # Implementation of Thue-Morse
    t = np.zeros(N, dtype=int)
    for i in range(N):
        # Count bits set in i
        count = 0
        temp = i
        while temp:
            count += temp & 1
            temp >>= 1
        t[i] = count % 2
        
    # Map 0 to 1, 1 to -1
    # t[i] = 0 -> 1
    # t[i] = 1 -> -1
    heights = 1 - 2 * t
    
    return heights
# EVOLVE-BLOCK-END
