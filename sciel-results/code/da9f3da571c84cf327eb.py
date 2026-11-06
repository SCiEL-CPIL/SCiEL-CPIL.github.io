import numpy as np
import time
import random

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _greedy_pass(primes_order, R):
    # R is passed as a fresh zero array
    current_S_size = N + 1
    for p in primes_order:
        if p > current_S_size:
            break
        
        best_r = 0
        min_new = N + 1
        
        # Vectorized counting for each residue
        # We can compute counts for all residues at once?
        # indices = np.arange(N+1)
        # residues = indices % p
        # available = ~R
        # counts = np.bincount(residues[available], minlength=p)
        # This is O(N) per prime.
        
        indices = np.arange(N + 1)
        residues = indices % p
        available_mask = ~R
        # Only count available elements
        avail_indices = indices[available_mask]
        if len(avail_indices) == 0:
            break
            
        avail_residues = avail_indices % p
        counts = np.bincount(avail_residues, minlength=p)
        
        # Find residue with min count
        # If a residue has 0 count, it's the best (removes nothing new)
        # But we must remove a class. If count is 0, we remove 0 new elements.
        # Wait, the condition is "some residue class is missed".
        # This means we choose a residue r to REMOVE.
        # The elements removed are those x in S such that x % p == r.
        # We want to minimize the number of NEW elements removed.
        # So we want to minimize the count of available elements in that residue class.
        
        min_count = np.min(counts)
        # If there are multiple, pick any. 
        # To be deterministic or random, we can pick the first one or random.
        # Let's pick the first one with min count.
        best_r = np.argmin(counts)
        
        # Update R
        R[indices[residues == best_r]] = True
        
        # Update S size
        # current_S_size -= min_count
        # Actually, we can just compute at the end or maintain a counter.
        # For the condition p > current_S_size, we need the current size.
        # current_S_size = (N+1) - sum(R)
        # This sum is O(N). Doing it every step is O(N * pi(N)) which is fine.
        # But we can update incrementally.
        # new_removed = min_count
        # current_S_size -= new_removed
        
    return (N + 1) - np.sum(R)

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    primes = _primes(N + 1)
    
    best_size = 0
    best_R = np.zeros(N + 1, dtype=bool)
    
    # Strategy 1: Ascending
    # Strategy 2: Descending
    # Strategy 3: Random Shuffles
    
    orders = []
    orders.append(primes) # Ascending
    orders.append(list(reversed(primes))) # Descending
    
    # Add some random shuffles
    # How many? Each pass is ~3.3M ops. 100 passes is 330M ops.
    # NumPy bincount is fast. Let's say 10-20 us per prime?
    # 669 primes * 20us = 13ms per pass.
    # 100 passes = 1.3s.
    # We have 290s. We can do many more.
    # Let's do 500 random shuffles.
    
    for _ in range(500):
        if time.time() > deadline - 10: # Safety margin
            break
        shuffled = primes.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    for order in orders:
        if time.time() > deadline - 5:
            break
            
        R = np.zeros(N + 1, dtype=bool)
        size = _greedy_pass(order, R)
        
        if size > best_size:
            best_size = size
            best_R = R.copy()
            
    points = np.nonzero(~best_R)[0].tolist()
    return points
