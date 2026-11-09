import time
import sys

N = 5000
DEADLINE = 320

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    
    all_primes = _primes(N + 1)
    prime_set = set(all_primes)
    
    # We will maintain residue counts for all primes up to N
    # But we only care about primes <= current_size + 1
    # Let's pre-allocate counts for all primes to avoid dynamic resizing issues
    # counts[p] is a list of length p
    
    # To speed up, we only need to track missing counts for primes <= N
    # missing[p] = number of residues r in [0, p-1] such that count[p][r] == 0
    
    # Initialize
    # We start with empty set
    # points = set()
    # But we will iterate x from 0 to N
    
    # For fast access, we can use a list of booleans for "is in set"
    in_set = [False] * (N + 1)
    
    # Counts structure
    # Using a dict or list of lists
    # primes are sparse, so dict is fine, but list of lists indexed by prime value is faster if we map prime to index?
    # No, prime values are up to 5000. We can just have an array `residue_counts` of size N+1, where residue_counts[p] is None if p is not prime, else a list of p zeros.
    
    residue_counts = [None] * (N + 1)
    missing_counts = [0] * (N + 1)
    
    for p in all_primes:
        residue_counts[p] = [0] * p
        missing_counts[p] = p
        
    current_size = 0
    
    # Candidates
    # We can try to add in increasing order
    # To potentially get a larger set, we might want to prioritize numbers that are "less restrictive"?
    # Actually, any number is equally restrictive in terms of "filling a gap".
    # Let's just iterate 0..N
    
    # Optimization: Precompute x % p for all x and p?
    # That's 5000 * 5000 = 25M operations. Might be too slow in Python.
    # Better to compute on the fly.
    
    # We need to check primes p <= current_size + 1
    # We can keep a list of active primes
    active_primes = []
    next_prime_idx = 0
    
    for x in range(N + 1):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Determine which primes to check
        # Primes p such that p <= current_size + 1
        # We need to add new primes to active_primes if current_size+1 crosses a prime
        
        target_size = current_size + 1
        
        while next_prime_idx < len(all_primes) and all_primes[next_prime_idx] <= target_size:
            active_primes.append(all_primes[next_prime_idx])
            next_prime_idx += 1
            
        # Check if x can be added
        valid = True
        for p in active_primes:
            if missing_counts[p] == 1:
                r = x % p
                if residue_counts[p][r] == 0:
                    valid = False
                    break
                    
        if valid:
            in_set[x] = True
            current_size += 1
            
            # Update counts
            for p in all_primes:
                # We only need to update counts for primes that might become active or are already active?
                # Actually, we need counts for ALL primes to be ready for future steps
                # But updating counts for all primes for every x is O(pi(N)) per x.
                # pi(5000) ~ 670.
                # 5000 * 670 = 3.35M operations. This is fine.
                
                r = x % p
                if residue_counts[p][r] == 0:
                    missing_counts[p] -= 1
                residue_counts[p][r] += 1
                
    return [i for i in range(N + 1) if in_set[i]]
