"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    N = 123457
    
    # Compute Largest Prime Factor for each number up to N using NumPy vectorization
    # This is significantly faster than pure Python loops for this size
    lpf = np.ones(N + 1, dtype=np.int32)
    
    # Sieve for largest prime factor
    # Iterate through potential primes
    # If lpf[i] == 1, i is prime
    for i in range(2, N + 1):
        if lpf[i] == 1:
            # Mark all multiples of i with i as their largest prime factor
            # Since we iterate i in increasing order, later (larger) primes
            # will overwrite earlier (smaller) primes in lpf[j]
            lpf[i::i] = i
            
    # Vectorized scan for the optimal decomposition
    # a ranges from 1 to N-1
    a = np.arange(1, N, dtype=np.int32)
    b = N - a
    
    pa = lpf[a]
    pb = lpf[b]
    
    # Compute max(P(a), P(b)) for all pairs
    m = np.maximum(pa, pb)
    
    # Find the index with the minimum max prime factor
    idx = np.argmin(m)
    
    best_a = int(a[idx])
    best_b = int(b[idx])
    
    # Check if we hit the theoretical lower bound (both powers of 2, so P=2 or 1)
    # If best_max_pf is 1, it means both are 1 (impossible since a+b=123457 > 2)
    # If best_max_pf is 2, it means both are powers of 2.
    # Since 123457 is odd, one is even, one is odd.
    # If both are powers of 2, the odd one must be 1.
    # So a=1, b=123456 or a=123456, b=1.
    # P(1)=1, P(123456) = ? 123456 = 2^4 * 3 * 11 * 13 * 19? Let's check.
    # 123456 / 16 = 7716
    # 7716 / 4 = 1929
    # 1929 / 3 = 643
    # 643 is prime.
    # So P(123456) = 643.
    # Thus max(P(1), P(123456)) = 643.
    # The vectorized scan will find the true minimum.
    
    # Return the result
    return [best_a, best_b]

# EVOLVE-BLOCK-END
