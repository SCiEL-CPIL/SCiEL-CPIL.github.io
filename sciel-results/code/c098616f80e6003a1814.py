"""Rewritten program using segmented sieve for largest prime factor and linear scan."""
import math

# EVOLVE-BLOCK-START
def construct():
    N = 123457
    # Compute smallest prime factor and largest prime factor using linear sieve
    spf = [0] * N  # smallest prime factor
    lpf = [1] * N  # largest prime factor
    primes = []
    
    for i in range(2, N):
        if spf[i] == 0:
            spf[i] = i
            lpf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p >= N:
                break
            spf[i * p] = p
            lpf[i * p] = p if p > lpf[i] else lpf[i]
    
    # Find the minimum of max(lpf[a], lpf[N-a]) for a in [1, N-1]
    best_val = float('inf')
    best_a = 1
    
    for a in range(1, N):
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
    
    a = best_a
    b = N - a
    return [a, b]
# EVOLVE-BLOCK-END
