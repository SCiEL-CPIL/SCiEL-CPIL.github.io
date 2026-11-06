import time
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: 90s limit, use 85s for safety
    deadline = start_time + 85.0
    
    LO = 10000
    HI = 1000000
    MAX_VAL = HI + 1
    
    # Precompute Largest Prime Factor (LPF) for all numbers up to MAX_VAL
    # lpf[i] will hold the largest prime factor of i
    # Initialize with 0. For prime p, lpf[p] remains 0 until we process p,
    # but for composite numbers, it will be updated by their prime factors.
    # Wait, if we just assign lpf[j] = p, the last assignment wins.
    # Since we iterate p in increasing order, the largest prime factor will be the last one assigned.
    lpf = [0] * (MAX_VAL + 1)
    
    # Sieve of Eratosthenes variant for LPF
    for p in range(2, MAX_VAL + 1):
        if lpf[p] == 0:
            # p is prime
            # Mark all multiples of p with p as a prime factor
            # Since we go in increasing order, the last assignment will be the largest prime factor
            for multiple in range(p, MAX_VAL + 1, p):
                lpf[multiple] = p
                
    # Check time
    if time.time() > deadline:
        # Fallback to a simple heuristic if sieve was too slow (unlikely for 1e6)
        # But we can return a known good value or LO
        return LO

    # Convert to numpy for fast vectorized max and argmin
    # We need lpf[LO] to lpf[HI] and lpf[LO+1] to lpf[HI+1]
    lpf_arr = np.array(lpf, dtype=np.int32)
    
    # Extract relevant slices
    # n ranges from LO to HI
    # n+1 ranges from LO+1 to HI+1
    lpf_n = lpf_arr[LO:HI+1]
    lpf_n1 = lpf_arr[LO+1:HI+2]
    
    # Compute the cost for each n: max(lpf[n], lpf[n+1])
    costs = np.maximum(lpf_n, lpf_n1)
    
    # Find the index of the minimum cost
    min_idx = np.argmin(costs)
    
    # The corresponding n is LO + min_idx
    best_n = LO + min_idx
    
    return int(best_n)
