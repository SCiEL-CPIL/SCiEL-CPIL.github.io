"""Improved program using Sieve for Largest Prime Factors."""
import time

def construct():
    target = 123457
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin for 90s limit
    
    # Precompute largest prime factor for all numbers up to target
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (target + 1)
    lpf[1] = 1
    
    # Sieve to find largest prime factors
    # We iterate i from 2 to target.
    # If lpf[i] is 0, i is a prime number.
    # For each prime i, we update lpf[j] = i for all multiples j of i.
    # Since we iterate i in increasing order, the last assignment to lpf[j]
    # will be by the largest prime factor of j.
    for i in range(2, target + 1):
        if time.time() > deadline:
            break
        if lpf[i] == 0:  # i is prime
            for j in range(i, target + 1, i):
                lpf[j] = i
                
    best_a = 1
    best_b = target - 1
    # Initialize best_score with a valid split if sieve was interrupted or just standard init
    if lpf[1] == 0: # Should not happen, but safe
        best_score = float('inf')
    else:
        best_score = max(lpf[1], lpf[target - 1]) if lpf[target-1] > 0 else float('inf')
        
    # Iterate through all possible splits
    # We can optimize by symmetry: checking a and b is same as b and a.
    # So we only need to check a up to target // 2.
    half_target = target // 2
    
    for a in range(2, half_target + 1):
        if time.time() > deadline:
            break
        b = target - a
        # Ensure lpf values are computed. If sieve was interrupted, some lpf[b] might be 0.
        # If lpf[b] is 0, it means b is prime (if sieve reached b) or sieve didn't reach b.
        # If sieve didn't reach b, we can't trust lpf.
        # However, for N=123457, the sieve is very fast (<0.1s).
        # We assume the sieve completes.
        if lpf[a] == 0 or lpf[b] == 0:
            continue
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
    return [best_a, best_b]
