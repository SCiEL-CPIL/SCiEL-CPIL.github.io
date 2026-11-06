"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find two positive integers a, b with a+b=123457 minimizing max(P(a), P(b)),
    where P(n) is the largest prime factor of n (P(1)=1).
    
    Strategy:
    1. Precompute the largest prime factor (LPF) for all numbers up to N=123457
       using a sieve-like approach.
    2. Iterate through all possible splits a from 1 to N//2.
    3. For each a, compute b = N - a.
    4. Compute cost = max(LPF[a], LPF[b]).
    5. Track the minimum cost and the corresponding (a, b).
    """
    start_time = time.time()
    deadline = 60.0  # seconds
    
    N = 123457
    
    # Precompute largest prime factor for all numbers up to N
    # Using a sieve approach: for each prime p, mark multiples of p with p as a factor
    # We want the LARGEST prime factor, so we iterate primes in increasing order
    # and overwrite, so the last write is the largest prime.
    
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    # Sieve for largest prime factor
    # For each i from 2 to N, if lpf[i] == 0, then i is prime
    # Mark all multiples of i with i
    
    # This is O(N log log N) roughly, but actually each composite is marked by each of its prime factors
    # Wait, this approach marks each multiple of i. The total work is sum_{p prime <= N} N/p = N * sum_{p} 1/p ~ N log log N.
    # This is efficient.
    
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    # Now find the optimal split
    best_cost = float('inf')
    best_a = 1
    best_b = N - 1
    
    # We only need to check a from 1 to N//2, since (a,b) and (b,a) are symmetric
    # But we must ensure a >= 1 and b >= 1, so a from 1 to N-1
    # By symmetry, a from 1 to N//2 is sufficient, and we return [a, b] or [b, a]
    
    half = N // 2
    for a in range(1, half + 1):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        cost = max(pa, pb)
        if cost < best_cost:
            best_cost = cost
            best_a = a
            best_b = b
        
        # Check time periodically
        if a % 10000 == 0:
            if time.time() > deadline - 5:
                break
    
    # Return the pair
    return [best_a, best_b]
# EVOLVE-BLOCK-END
