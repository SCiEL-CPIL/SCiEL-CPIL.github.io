"""
Optimized smooth two-summand decomposition.
Uses a segmented-style sieve to compute the largest prime factor for all integers up to TARGET,
then performs a linear scan to find the pair (a, b) with a+b=TARGET minimizing max(P(a), P(b)).
"""
import time
import math

def construct():
    deadline = time.time() + 50  # Safety margin before 60s limit
    TARGET = 123457
    
    # Sieve to compute largest prime factor (LPF) for all numbers up to TARGET
    # lpf[i] stores the largest prime factor of i
    lpf = [0] * (TARGET + 1)
    lpf[1] = 1
    
    # Standard sieve modification: for each prime p, mark its multiples with p
    # Since we iterate p in increasing order, the last prime that divides a number
    # will be the largest one, so we just overwrite lpf[j] = p.
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, TARGET + 1, i):
                lpf[j] = i
                
    # Now lpf[n] contains the largest prime factor for all n in [1, TARGET]
    # We need to find a in [1, TARGET-1] such that b = TARGET - a
    # and max(lpf[a], lpf[b]) is minimized.
    
    best_val = float('inf')
    best_pair = [1, TARGET - 1]
    
    # Iterate a from 1 to TARGET//2 (symmetry: checking a and TARGET-a covers all pairs)
    # Note: if a == b, then TARGET must be even, but 123457 is odd, so a != b always.
    half = TARGET // 2
    for a in range(1, half + 1):
        b = TARGET - a
        # max(lpf[a], lpf[b])
        va = lpf[a]
        vb = lpf[b]
        val = va if va > vb else vb
        
        if val < best_val:
            best_val = val
            best_pair = [a, b]
            
            # Early termination check:
            # The minimum possible value for max(P(a), P(b)) is at least 2.
            # However, since TARGET is odd, one of a, b is even and the other is odd.
            # If the even number is a power of 2, its P is 2.
            # If the odd number is a power of 3, its P is 3.
            # So max could be 3?
            # Let's check if we can do better than 3.
            # If best_val == 2, it's impossible for odd sum (one is even, one odd. If even is 2^k, P=2. If odd is 1, P=1. But 1+2^k=123457? 123456 is not a power of 2).
            # Actually, if best_val becomes very small, we can stop?
            # The theoretical minimum for an odd sum N > 3 is likely 3 or 5 or similar.
            # But without knowing the exact lower bound, we can't stop early safely unless we prove optimality.
            # However, the loop is O(N) which is ~60k iterations, very fast.
            # The sieve is O(N log log N) which is also fast.
            # So we just let it finish to guarantee global optimum.
            
        # Optional: time check
        if time.time() > deadline:
            break
    
    return best_pair

# EVOLVE-BLOCK-END
