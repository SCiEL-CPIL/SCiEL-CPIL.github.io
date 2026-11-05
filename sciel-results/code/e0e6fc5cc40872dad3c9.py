import time
from math import isqrt

def construct():
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    # Step 1: Generate all primes up to TARGET using sieve
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(TARGET) + 1):
        if is_prime[i]:
            for j in range(i*i, TARGET + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, TARGET + 1) if is_prime[i]]
    
    # Step 2: For each prime K (as candidate max prime factor), generate K-smooth numbers
    # and check if any two sum to TARGET
    # A K-smooth number has all prime factors <= K
    
    def generate_smooth_numbers(limit, max_prime, primes):
        """Generate all numbers <= limit whose prime factors are all <= max_prime"""
        smooth = set()
        # Use DFS/backtracking to generate smooth numbers
        def dfs(idx, current):
            if current > limit:
                return
            smooth.add(current)
            for i in range(idx, len(primes)):
                if primes[i] > max_prime:
                    break
                if current * primes[i] > limit:
                    break
                dfs(i, current * primes[i])
        
        dfs(0, 1)
        return smooth
    
    # Start with small primes and increase K
    # We want to find the smallest K such that there exist a+b=TARGET with P(a),P(b) <= K
    
    # First, let's try a direct approach: scan all a and compute P(a), P(b)
    # but do it efficiently by precomputing largest prime factor using a sieve-like method
    
    # Precompute largest prime factor using modified sieve
    lpf = [0] * (TARGET + 1)
    lpf[1] = 1
    
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, TARGET + 1, i):
                lpf[j] = i  # Since we go in increasing order, last assignment is largest
    
    # Now scan all pairs
    best_max_p = TARGET
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET // 2 + 1):
        b = TARGET - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = max(pa, pb)
        
        if current_max < best_max_p:
            best_max_p = current_max
            best_a = a
            best_b = b
            
            # Early termination: if we found a solution with max prime factor 
            # that's quite small, we can stop
            # But let's continue to ensure optimality within time limit
        
        if time.time() - start_time > DEADLINE:
            break

    return [best_a, best_b]
