import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Fast phi function
    def phi(x):
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                while y % p == 0:
                    y //= p
                result -= result // p
        if y > 1:
            result -= result // y
        return result

    # 3. Determine search range
    # We want to search as far as possible.
    # Let's start with 5,000,000 and see if we can do more.
    # Actually, let's just iterate and check time.
    # We need a buffer for the window length k (max 200).
    
    best_k = 1
    best_s = 1
    
    # We will compute phi on the fly for a sliding window?
    # No, precomputing a list is faster for the distinct check.
    # But we don't know how far we can go.
    # Let's compute in chunks.
    
    # Strategy: 
    # Iterate s from 1 upwards.
    # Maintain a list of recent phi values.
    # For each new s, check the run starting at s.
    # This is inefficient if done naively.
    
    # Better:
    # Compute phi for 1..N.
    # Then scan for best run.
    # Let's try N = 2,000,000 first.
    # If time allows, we can't easily extend a precomputed list without recomputing.
    # So we pick a fixed N that fits in time.
    # 2e6 phi calculations in Python:
    # ~2e6 * 20 ops = 4e7 ops. Should be < 10s.
    # Let's try N = 5,000,000.
    
    N = 5000000
    # Check if we have enough time for 5e6
    # Estimate: 5e6 phi calls.
    # Let's just run it and break if time is up.
    
    phis = [0] * (N + 201)
    
    for i in range(1, N + 201):
        if time.time() > deadline:
            # If we run out of time, shrink N to what we have
            N = i - 1
            break
        phis[i] = phi(i)
        
    if N < 1:
        return {'start': 1, 'length': 1}
        
    # Now find the best run in phis[1..N+200]
    # We only care about starts s such that s + k - 1 <= N + 200
    # And s <= N (we computed up to N+200, but starts should be <= N to be "safe" in the main range?)
    # The task says s + k - 1 <= 10^12.
    # Our computed range is much smaller.
    # We just need to find the best run within the computed array.
    
    # Optimization: 
    # For each s, extend k.
    # We can use a sliding window distinct count?
    # No, we need the MAX k starting at s.
    # So for each s, we expand until collision.
    
    # To speed up, we can skip s if we already know a run of length L starts at s-1?
    # Not necessarily.
    
    # Simple loop:
    for s in range(1, N + 1):
        if time.time() > deadline:
            break
        seen = set()
        k = 0
        # Max k is 200
        while s + k <= N + 200 and k < 200:
            val = phis[s + k]
            if val in seen:
                break
            seen.add(val)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            if best_k >= 200:
                break

    return {'start': best_s, 'length': best_k}
