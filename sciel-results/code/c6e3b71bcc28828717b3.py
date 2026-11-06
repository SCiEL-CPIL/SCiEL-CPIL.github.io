import numpy as np
import time

N = 5000

def _primes(limit):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p:limit+1:p] = False
    return np.where(sieve)[0]

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_set = []
    
    # Strategy 1: Coprimes to P_max#
    # P_max from 30 to 150
    for P_max in range(30, 150, 5):
        if time.time() > deadline:
            break
            
        # Generate candidates: x in [0, N] such that gcd(x, product_primes_le_Pmax) == 1
        # Actually, just check x % p != 0 for all p <= P_max
        primes_to_check = _primes(P_max)
        
        # Vectorized check
        x = np.arange(N + 1)
        mask = np.ones(N + 1, dtype=bool)
        for p in primes_to_check:
            mask[x % p == 0] = False
            
        candidates = np.where(mask)[0]
        
        # Repair
        points = candidates.tolist()
        changed = True
        while changed and time.time() < deadline:
            changed = False
            k = len(points)
            # Only need to check primes <= k
            primes_to_repair = _primes(k)
            for p in primes_to_repair:
                # Check if all residues covered
                residues = np.array([v % p for v in points])
                unique_res = np.unique(residues)
                if len(unique_res) == p:
                    # Find least populated
                    counts = np.bincount(residues, minlength=p)
                    min_res = np.argmin(counts)
                    # Remove
                    points = [v for v in points if v % p != min_res]
                    changed = True
                    break
                    
        if len(points) > len(best_set):
            best_set = points

    # Strategy 2: Random sets
    # Try a few random sets of size ~ 500
    for _ in range(50):
        if time.time() > deadline:
            break
        # Random sample
        points = np.random.choice(N + 1, size=600, replace=False).tolist()
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            k = len(points)
            primes_to_repair = _primes(k)
            for p in primes_to_repair:
                residues = np.array([v % p for v in points])
                unique_res = np.unique(residues)
                if len(unique_res) == p:
                    counts = np.bincount(residues, minlength=p)
                    min_res = np.argmin(counts)
                    points = [v for v in points if v % p != min_res]
                    changed = True
                    break
                    
        if len(points) > len(best_set):
            best_set = points

    return best_set
