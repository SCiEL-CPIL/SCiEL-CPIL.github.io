import numpy as np
import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # R is a boolean array where R[x] is True if x is removed
    R = np.zeros(N + 1, dtype=bool)
    
    primes = _primes(N + 1)
    
    for p in primes:
        if time.time() > deadline:
            break
            
        current_S_size = (N + 1) - np.sum(R)
        
        # We only need to satisfy the condition for primes p <= |S|
        # If p > |S|, we can stop
        if p > current_S_size:
            break
            
        # Find the residue r that minimizes the number of NEW elements removed
        # New elements are those x such that x % p == r and R[x] is False
        
        best_r = 0
        min_new = N + 1
        
        # We can vectorize the counting for speed
        # Create an array of indices 0..N
        # For each r, mask = (indices % p == r) & (~R)
        # Count = sum(mask)
        
        # To be faster, we can iterate r and use numpy slicing if possible?
        # x % p == r is not a simple slice.
        # But we can precompute indices for each r?
        # Indices for r: r, r+p, r+2p...
        
        # Let's just loop r, it's small (p <= 5000, but usually small primes dominate the cost)
        # Actually, for large p, p is large, but N/p is small.
        
        for r in range(p):
            # Generate indices
            # np.arange(r, N+1, p)
            # Check which are not in R
            # indices = np.arange(r, N + 1, p)
            # new_count = np.sum(~R[indices])
            
            # Optimization: 
            # We can maintain a count of available numbers in each residue class?
            # But R changes.
            
            # Let's stick to the simple loop first, it should be fast enough.
            # 5000 / p iterations per r.
            # Sum over r is 5000.
            # So for each p, we do ~5000 operations.
            # 669 primes * 5000 = 3.3M ops. Very fast.
            
            indices = np.arange(r, N + 1, p)
            new_count = np.count_nonzero(~R[indices])
            
            if new_count < min_new:
                min_new = new_count
                best_r = r
                
        # Update R
        indices = np.arange(best_r, N + 1, p)
        R[indices] = True
        
    # The admissible tuple is the set of x where R[x] is False
    # We need to return a list of distinct integers
    points = np.nonzero(~R)[0].tolist()
    
    return points
