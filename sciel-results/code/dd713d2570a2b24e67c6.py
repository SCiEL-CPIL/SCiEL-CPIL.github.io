"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 123457
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We want to minimize max(P(a), P(b)).
    # This is equivalent to finding the smallest prime p such that there exist a, b
    # with a+b=N and P(a)<=p, P(b)<=p.
    # We can iterate over primes p and check if a solution exists using p-smooth numbers.
    
    # To be efficient, we can generate p-smooth numbers for increasing p.
    # However, the set of p-smooth numbers grows.
    # Let's try small primes first.
    
    best_a, best_b = 1, N - 1
    best_val = float('inf')
    
    # Precompute powers for speed? Or just recursive generation.
    # Recursive generation of smooth numbers up to N.
    
    def generate_smooth(primes_subset, limit):
        smooth = set()
        def gen(idx, current):
            if current > limit:
                return
            if idx == len(primes_subset):
                smooth.add(current)
                return
            p = primes_subset[idx]
            # Multiply by p^k
            val = current
            while val <= limit:
                gen(idx + 1, val)
                val *= p
        gen(0, 1)
        return smooth

    # Try increasing sets of primes
    # We need to check if there is a pair in the smooth set summing to N.
    # We can just check all pairs or use a set.
    
    # Since N is fixed, we can just iterate through the smooth set and check if N-x is in the set.
    
    # Let's try k=1 (primes [2]), k=2 (primes [2,3]), etc.
    # Actually, we can just generate all smooth numbers for the first few primes and check.
    # If we find a pair, the max prime factor is the largest prime in the subset.
    
    # Optimization: The number of smooth numbers grows.
    # For p=2: powers of 2.
    # For p=3: 2^a 3^b.
    # ...
    
    # Let's iterate through primes and add them one by one, checking for solution.
    # But the set changes.
    
    # Alternative: Just generate all numbers up to N and compute P(n) using a sieve, 
    # but that's what the original code did.
    
    # Why did the original code score 58?
    # Maybe the optimal solution has max prime factor ~58?
    # Or maybe the original code didn't finish? 
    # The original code has a deadline check.
    # If it didn't finish, it returns the best found so far.
    # If it did finish, it returns the global minimum.
    
    # Let's assume the original code finished and found 58.
    # Can we do better?
    # Let's check if a solution with smaller max prime factor exists.
    
    # Let's try the smooth number approach with a cap on the number of smooth numbers generated to avoid explosion.
    
    # Generate smooth numbers for primes [2, 3, 5, 7, 11, 13, 17, 19, 23]
    # Check if any pair sums to N.
    
    # We can do this incrementally.
    # Start with S = {1}.
    # For each prime p in [2, 3, 5, ...]:
    #   S_new = { x * p^k for x in S, k>=0, x*p^k <= N }
    #   Check if there is a, b in S_new such that a+b=N.
    #   If yes, return a, b.
    #   S = S_new
    
    # This is efficient because S grows slowly for small primes.
    
    S = {1}
    # We need to keep the set sorted or just use a set for lookup.
    # For checking a+b=N, we can iterate a in S and check if N-a in S.
    
    for p in primes:
        if time.time() - start_time > DEADLINE:
            break
            
        # Generate new smooth numbers
        S_new = set(S)
        for x in list(S):
            val = x
            while val <= N:
                S_new.add(val)
                val *= p
        
        # Check for solution
        found = False
        for a in S_new:
            b = N - a
            if b in S_new:
                # Found a pair
                # The max prime factor is at most p (actually could be less if a,b don't use p)
                # But we are iterating p in increasing order, so this is the minimal max prime factor 
                # IF we ensure that we only add p when necessary? 
                # No, S_new contains numbers that might not have p as a factor.
                # However, if a solution existed with max prime factor < p, it would have been found in a previous iteration.
                # So the first time we find a solution, the max prime factor is exactly the current p 
                # (or possibly smaller, but we would have found it earlier).
                # Wait, if a solution exists with max prime factor 7, we find it when p=7.
                # If a solution exists with max prime factor 5, we find it when p=5.
                # So the first p for which we find a solution gives the optimal max prime factor.
                
                # We need to return the pair.
                # We should verify that P(a) <= p and P(b) <= p.
                # By construction, all numbers in S_new are p-smooth, so their prime factors are in {2, ..., p}.
                # So P(a) <= p and P(b) <= p.
                
                return [a, b]
        
        S = S_new
        
    # If no solution found with these primes, fallback to the original scan
    # But likely a solution is found early.
    
    # Fallback
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N // 2 + 1):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        m = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
        if best_max_pf == 1:
            break
            
    return [best_a, best_b]

# EVOLVE-BLOCK-END
