"""Improved program using Sieve for Largest Prime Factors."""
import time

def construct():
    target = 123457
    
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
        if lpf[i] == 0:  # i is prime
            for j in range(i, target + 1, i):
                lpf[j] = i
                
    best_a = 1
    best_b = target - 1
    best_score = max(lpf[best_a], lpf[best_b])
    
    # Iterate through all possible splits
    # We can optimize by symmetry: checking a and b is same as b and a.
    # So we only need to check a up to target // 2.
    half_target = target // 2
    
    for a in range(2, half_target + 1):
        b = target - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
    return [best_a, best_b]
