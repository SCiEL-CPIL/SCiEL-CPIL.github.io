"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    N_MIN = 10000
    N_MAX = 1000000
    
    # We need LPF for numbers up to N_MAX + 1 (since we check n+1)
    limit = N_MAX + 1
    
    # Initialize array to hold largest prime factor
    # lpf[i] will store the largest prime factor of i
    # Initialize with 0 or 1. Since we iterate primes in increasing order,
    # the last prime that divides i will be the largest one.
    lpf = [0] * (limit + 1)
    
    # Sieve for largest prime factor
    # Iterate through all potential primes
    # If lpf[i] is 0, i is prime (no smaller prime has divided it yet in a way that sets its LPF? 
    # Actually, standard sieve: if we iterate p from 2 to limit, and for each p, 
    # if lpf[p] == 0, then p is prime. Then for all multiples m of p, set lpf[m] = p.
    # Since we go from small p to large p, the final value in lpf[m] will be the largest prime factor.
    
    for p in range(2, limit + 1):
        if lpf[p] == 0:
            # p is prime
            for m in range(p, limit + 1, p):
                lpf[m] = p
                
    # Now lpf[i] contains the largest prime factor of i for all i in [2, limit]
    # lpf[1] is 0, but 1 has no prime factors. We can treat LPF(1) as 1 for max purposes, or just handle it.
    # The problem asks for LPF of n(n+1). n >= 10000, so n and n+1 are > 1.
    
    best_lpf = float('inf')
    best_n = N_MIN
    
    # Scan the range [N_MIN, N_MAX]
    for n in range(N_MIN, N_MAX + 1):
        lpf_n = lpf[n]
        lpf_np1 = lpf[n + 1]
        
        # The largest prime factor of n(n+1) is max(LPF(n), LPF(n+1))
        current_max_lpf = max(lpf_n, lpf_np1)
        
        if current_max_lpf < best_lpf:
            best_lpf = current_max_lpf
            best_n = n
            
    return best_n

# EVOLVE-BLOCK-END
