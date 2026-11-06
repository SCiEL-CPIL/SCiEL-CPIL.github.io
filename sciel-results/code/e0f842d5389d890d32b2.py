import numpy as np
import time

def construct():
    deadline = time.time() + 85
    LO = 10000
    HI = 1000000
    MAX_VAL = HI + 1
    
    # Initialize lpf array
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    # Sieve of Eratosthenes to find primes and compute LPF
    # We iterate i from 2 to MAX_VAL
    # If lpf[i] is 0, i is prime.
    # For all multiples j of i, set lpf[j] = i.
    # Since i increases, the final lpf[j] is the largest prime factor.
    
    # Optimization: Only need to iterate i up to MAX_VAL
    # But checking lpf[i]==0 is fast.
    
    for i in range(2, MAX_VAL + 1):
        if lpf[i] == 0:
            # i is prime
            # Update multiples
            # lpf[i::i] = i
            lpf[i::i] = i
            
    # Compute max(lpf[n], lpf[n+1]) for n in [LO, HI]
    # lpf[LO:HI+1] corresponds to n
    # lpf[LO+1:HI+2] corresponds to n+1
    # Note: HI+1 is MAX_VAL, so index MAX_VAL is valid.
    # lpf array size is MAX_VAL + 1, indices 0..MAX_VAL.
    # We need lpf[n+1] for n=HI, which is lpf[HI+1] = lpf[MAX_VAL].
    
    vals = np.maximum(lpf[LO:HI+1], lpf[LO+1:HI+2])
    
    # Find index of minimum
    min_idx = np.argmin(vals)
    best_n = LO + min_idx
    
    # Return as Python int
    return int(best_n)
