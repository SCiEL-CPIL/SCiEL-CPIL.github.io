import time
import random
import math

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    best_size = 0
    best_points = []
    
    # We will run multiple randomized greedy attempts
    # The greedy algorithm:
    # 1. active = [True] * (N + 1)
    # 2. For p in PRIMES:
    #    a. Count active in each class mod p
    #    b. Find class(es) with min count
    #    c. Randomly pick one if ties
    #    d. Set active[x] = False for x in that class
    
    # Optimization: We only need to check primes up to current active count?
    # The condition is p <= len(tuple).
    # As we remove elements, len(tuple) decreases.
    # So we can stop early if p > current_size.
    # But we need to know current_size.
    
    while time.time() < deadline:
        active = [True] * (N + 1)
        current_size = N + 1
        
        # Process primes
        # We need to process primes in increasing order?
        # Yes, because if we skip a small prime, we might remove too many later?
        # Actually, we MUST satisfy all p <= final_size.
        # If we process p=2, then p=3, etc.
        # We can break if p > current_size.
        
        for p in PRIMES:
            if p > current_size:
                break
            
            # Count active in each residue class
            counts = [0] * p
            for x in range(N + 1):
                if active[x]:
                    counts[x % p] += 1
            
            # Find min count
            min_count = min(counts)
            
            # Find all classes with min count
            candidates = [r for r in range(p) if counts[r] == min_count]
            
            # Randomly pick one
            r = random.choice(candidates)
            
            # Deactivate
            removed = 0
            for x in range(N + 1):
                if active[x] and x % p == r:
                    active[x] = False
                    removed += 1
            current_size -= removed
            
            # If current_size becomes 0, break
            if current_size == 0:
                break
        
        # Extract points
        points = [x for x in range(N + 1) if active[x]]
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
            
    return best_points
