"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    LO, HI = 10000, 1000000
    max_val = HI + 1  # we need up to HI+1 = 1000001
    
    # Precompute primes up to max_val using a standard sieve
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:max_val+1:step] = [False] * len(range(start, max_val+1, step))
    
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    
    # Segmented sieve for largest prime factor in [LO, max_val]
    # Size of segment
    size = max_val - LO + 1  # 1000001 - 10000 + 1 = 990002
    
    # Initialize LPF array to 1
    lpf = np.ones(size, dtype=np.int64)
    
    # For each prime p, find multiples of p in [LO, max_val] and update LPF
    # A number n in [LO, max_val] has LPF = max over all prime divisors p of n
    # For prime p, all multiples of p in the range should have p considered
    
    for p in primes:
        if p > max_val:
            break
        # Find the first multiple of p >= LO
        if p >= LO:
            first = p
        else:
            first = ((LO + p - 1) // p) * p
        
        if first > max_val:
            break
        
        # For each multiple of p in [first, max_val], step p
        # Update lpf if p is larger
        # Use numpy for vectorized update
        multiples = np.arange(first, max_val + 1, p)
        indices = multiples - LO  # indices into lpf array
        
        # Update lpf[indices] = max(lpf[indices], p)
        # Only update where p > current value
        mask = p > lpf[indices]
        if np.any(mask):
            lpf[indices[mask]] = p
    
    # Check time
    if time.time() > deadline - 10:
        # Fallback: just return LO (should not happen given the fast sieve)
        return LO
    
    # Compute max(lpf[n], lpf[n+1]) for n in [LO, HI]
    # lpf[i] corresponds to number LO + i
    # For n = LO + i, we need max(lpf[i], lpf[i+1]) for i in [0, HI-LO]
    # lpf has indices 0 to size-1, corresponding to LO to max_val
    # HI - LO = 990000, so we need i from 0 to 990000
    # lpf[0:990001] and lpf[1:990002] -> max element-wise
    
    # pair_max[i] = max(lpf[i], lpf[i+1]) for i in [0, HI-LO]
    # We need indices 0 to HI-LO inclusive, which is 0 to 990000
    # lpf[0:990001] has 990001 elements (indices 0..990000)
    # lpf[1:990002] has 990001 elements (indices 1..990001)
    
    n_elements = HI - LO + 1  # 990001
    pair_max = np.maximum(lpf[:n_elements], lpf[1:1+n_elements])
    
    # Find the minimum
    min_idx = np.argmin(pair_max)
    best_n = LO + min_idx
    best_val = int(pair_max[min_idx])
    
    return best_n
# EVOLVE-BLOCK-END
