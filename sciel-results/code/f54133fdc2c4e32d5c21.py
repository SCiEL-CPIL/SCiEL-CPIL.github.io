"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    N = 123457
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Compute largest prime factor (LPF) for all integers up to N
    # Using a modified sieve: for each prime p, update LPF for all multiples
    lpf = list(range(N + 1))  # lpf[i] = largest prime factor of i
    lpf[0] = 0
    if N >= 1:
        lpf[1] = 1
    
    # Sieve of Eratosthenes style: for each prime, mark its multiples
    # We only need to process primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    if N >= 1:
        is_prime[1] = False
    
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    # Now for each prime, update lpf for all its multiples
    for p in range(2, N + 1):
        if is_prime[p]:
            for j in range(p, N + 1, p):
                lpf[j] = p  # since we go in increasing order, last prime divisor is the largest
    
    # Now scan all pairs (a, N-a) for a in 1..N-1
    # We want to minimize max(lpf[a], lpf[N-a])
    best_val = float('inf')
    best_a = 1
    best_b = N - 1
    
    # Optimization: we can break early if we find a pair where both LPFs are small
    # But let's do a full scan since N is only ~123k
    for a in range(1, N):
        b = N - a
        m = max(lpf[a], lpf[b])
        if m < best_val:
            best_val = m
            best_a = a
            best_b = b
        # Early termination: if best_val == 2, that's the minimum possible (since both must be >=2 for a,b>1)
        # Actually if one is 1, LPF=1, but a,b must be positive integers, so a>=1, b>=1
        # If a=1, lpf[1]=1, lpf[N-1] could be large. Let's check if we can get max=2
        if best_val == 2:
            break
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
