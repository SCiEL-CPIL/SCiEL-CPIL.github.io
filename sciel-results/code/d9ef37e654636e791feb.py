import numpy as np
import time
import math

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin before 90s kill
    
    # 1. Sieve of Eratosthenes to find primes up to N
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i:N+1:i] = False
    
    primes = np.where(is_prime)[0]
    
    # 2. For each candidate M (prime), generate all M-smooth numbers in [1, N]
    # An M-smooth number has all prime factors ≤ M
    
    def generate_smooth_numbers(M, limit):
        """Generate all numbers in [1, limit] whose largest prime factor is ≤ M."""
        # Use a boolean array for membership
        smooth = np.zeros(limit + 1, dtype=bool)
        smooth[1] = True
        
        # For each prime p ≤ M, mark multiples
        for p in primes:
            if p > M:
                break
            # Mark all multiples of p that only have prime factors ≤ M
            # Actually, we need to be more careful: a number is M-smooth if ALL its prime factors are ≤ M
            # So we can't just mark multiples of individual primes
            
            # Better approach: use dynamic programming / BFS
            # Start with 1, and multiply by primes ≤ M
            pass
        
        # Alternative: for each number, check if it's M-smooth by factoring
        # But that's slow. Let's use a sieve-like approach.
        
        # A number n is M-smooth iff for every prime p > M, p does not divide n
        # Equivalently, n = g * h where g is M-smooth and h has no prime factors > M... that's circular.
        
        # Best approach: compute smallest prime factor, then check if max prime factor ≤ M
        # But we want to avoid O(N log N) per M.
        
        # Let's precompute LPF once, then for each M, check which numbers have LPF ≤ M
        return None
    
    # Precompute LPF for all numbers using a sieve approach
    lpf = np.ones(N + 1, dtype=np.int32)
    for p in primes:
        # For each prime p, mark p as the LPF for all multiples of p
        # But we want the LARGEST prime factor, so we process primes in increasing order
        # and overwrite - the last prime that divides n is the largest
        lpf[p::p] = p
    
    if time.time() > deadline:
        return (1, N - 1)
    
    # 3. Now iterate M through primes in increasing order
    # For each M, S_M = {x in [1, N] | lpf[x] ≤ M}
    # Check if there exists a in S_M such that N-a is also in S_M
    
    # Use a boolean array for fast membership
    in_S = np.zeros(N + 1, dtype=bool)
    
    # Sort numbers by their LPF to add them incrementally
    # Create an array of (lpf_value, number) pairs and sort by lpf_value
    numbers = np.arange(1, N + 1)
    lpf_vals = lpf[numbers]
    
    # Sort by lpf value
    sort_idx = np.argsort(lpf_vals, kind='stable')
    sorted_numbers = numbers[sort_idx]
    sorted_lpf = lpf_vals[sort_idx]
    
    # Iterate through unique LPF values in increasing order
    unique_lpf = np.unique(sorted_lpf)
    
    for M in unique_lpf:
        if time.time() > deadline:
            break
        
        # Add all numbers with lpf == M to S
        mask = (sorted_lpf == M)
        new_numbers = sorted_numbers[mask]
        
        for x in new_numbers:
            x = int(x)
            in_S[x] = True
            b = N - x
            if 1 <= b <= N and in_S[b]:
                return (x, b)
        
        # Also check: when we add x, maybe N-x was already in S
        # The above check handles this: if b is already in S, we return
    
    # Fallback: brute force scan
    a = np.arange(1, N, dtype=np.int32)
    b = N - a
    cost = np.maximum(lpf[a], lpf[b])
    min_idx = np.argmin(cost)
    return (int(a[min_idx]), int(b[min_idx]))
