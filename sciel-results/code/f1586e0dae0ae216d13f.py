import math
import random
import time

def construct():
    """Find the longest run of consecutive integers with distinct totients up to 10^12."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Step 1: Generate primes up to sqrt(10^12) = 10^6 for fast factorization
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Step 2: Fast phi computation using precomputed primes
    def phi(x):
        if x <= 1:
            return 1
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    # Step 3: Search strategy
    # Random sampling of large numbers exploits statistical rarity of totient collisions
    # at large scales, enabling discovery of long runs that exhaustive small-number search misses
    
    best_k = 1
    best_s = 1
    
    # First, try some structured locations where runs might be longer
    # These are heuristic starting points
    
    def check_run(s, max_k=200):
        """Check how long a run of distinct totients starts at s."""
        if s < 1 or s + max_k - 1 > 10**12:
            return 0
        seen = set()
        k = 0
        for i in range(max_k):
            n = s + i
            v = phi(n)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k
    
    # Try some promising starting points
    candidates = []
    
    # Small numbers (exhaustive up to a certain point)
    for s in range(1, 5000):
        candidates.append(s)
    
    # Structured candidates: products of primes, near powers, etc.
    # Numbers with many distinct prime factors tend to have smaller phi values,
    # but we want distinct phi values in a run
    
    # Try random large numbers
    random.seed(42)
    for _ in range(50000):
        # Sample from a wide range
        s = random.randint(10**9, 10**12 - 200)
        candidates.append(s)
    
    # Also try numbers around known "good" regions
    # Based on the observation that large numbers have rarer collisions
    for _ in range(30000):
        s = random.randint(10**10, 10**12 - 200)
        candidates.append(s)
    
    # Process candidates, checking time
    checked = set()
    for s in candidates:
        if time.time() > deadline:
            break
        if s in checked:
            continue
        checked.add(s)
        
        if s < 1 or s > 10**12 - 1:
            continue
        
        k = check_run(s)
        if k > best_k:
            best_k = k
            best_s = s
            # If we found a very long run, try to extend it
            # by checking neighbors
            for ds in range(-50, 51):
                if ds == 0:
                    continue
                ns = s + ds
                if ns < 1 or ns > 10**12 - 1:
                    continue
                if ns in checked:
                    continue
                checked.add(ns)
                nk = check_run(ns)
                if nk > best_k:
                    best_k = nk
                    best_s = ns
    
    # If we still have time, do a more systematic search in promising regions
    if time.time() < deadline - 10:
        # Focus on larger numbers where collisions are rarer
        step = 1000
        for s in range(10**9, min(10**12, int(10**11)), step):
            if time.time() > deadline:
                break
            if s in checked:
                continue
            checked.add(s)
            k = check_run(s)
            if k > best_k:
                best_k = k
                best_s = s
    
    return {'start': best_s, 'length': best_k}
