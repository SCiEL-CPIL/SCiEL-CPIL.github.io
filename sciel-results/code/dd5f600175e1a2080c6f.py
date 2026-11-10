import numpy as np
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
    # Estimate upper bound for K
    # K ~ N * exp(-gamma) / log(N) ? No, that's for prime gaps.
    # For admissible tuples, size K is roughly N * prod_{p<=K}(1-1/p).
    # prod_{p<=K}(1-1/p) ~ e^{-gamma} / log(K).
    # So K ~ N * e^{-gamma} / log(K).
    # For N=5000, e^{-gamma} ~ 0.56.
    # K * log(K) ~ 2800.
    # If K=400, 400 * 6 = 2400.
    # If K=500, 500 * 6.2 = 3100.
    # So K is around 450.
    
    # We will binary search for the largest K such that the greedy sieve leaves >= K elements.
    
    def greedy_sieve_size(K):
        # Returns the size of the set remaining after sieving with primes <= K
        # using the greedy min-count strategy.
        # We use a boolean mask for efficiency.
        mask = np.ones(N + 1, dtype=bool)
        
        # Only primes <= K are relevant
        primes_to_use = [p for p in PRIMES if p <= K]
        
        for p in primes_to_use:
            # Count elements in each residue class among those currently 'True'
            # We can use np.where to get indices, but that's slow if done naively.
            # Better: iterate through residues? No, N is small (5000).
            
            # Get indices of currently valid numbers
            # To make this fast, we can maintain a list of valid indices.
            # But with NumPy, we can just compute counts on the whole array masked.
            
            # Efficient way:
            # valid_indices = np.nonzero(mask)[0]
            # If len(valid_indices) == 0: return 0
            
            # For each residue r in 0..p-1:
            # count = sum(mask[i] for i in range(r, N+1, p))
            
            # This is O(N/p) per residue, total O(N) per prime.
            # Total O(N * pi(K)).
            
            # Let's try to vectorize the counting.
            # We can reshape or use modulo.
            # mods = np.arange(N+1) % p
            # This is O(N).
            
            # Actually, since N=5000 is small, a simple loop over valid indices in Python might be faster than NumPy overhead if the list gets small.
            # But the list starts at 5000.
            
            # Let's use a list of valid indices, which is what the current code does.
            # We can optimize the list operations.
            pass

    # The current code's logic is actually not that bad for N=5000.
    # The bottleneck is the repeated list filtering.
    # Let's just optimize the existing approach by:
    # 1. Using a list of valid indices.
    # 2. Using a dictionary or array for counts.
    # 3. Filtering efficiently.
    
    # Actually, let's look at the "Greedy Coloring" directive again.
    # "prioritizing vertices with the lowest degree in the remaining conflict graph"
    # This suggests a different algorithm:
    # 1. Build conflict graph? No, N=5000, edges ~ N^2 = 25M. Too big.
    # 2. The "conflict" is defined by the primes.
    
    # Let's stick to the sieve but make it faster.
    
    best_k = 0
    best_set = []
    
    # Scan K from estimated max down.
    # Estimate max K:
    # K_max = int(N * 0.6 / math.log(N)) + 100 # Rough guess
    # Let's just use 600 as a safe upper bound.
    
    for K in range(600, 0, -1):
        valid_indices = list(range(N + 1))
        
        # Primes <= K
        # Optimization: Precompute primes list up to 600
        # We have PRIMES.
        
        for p in PRIMES:
            if p > K:
                break
            
            if len(valid_indices) < K:
                # If we already have fewer than K elements, we can't form a set of size K
                # Wait, we need len(valid_indices) >= K at the end.
                # If it drops below K, it might come back? No, we only remove elements.
                # So if len < K, break early.
                break
            
            # Count residues
            counts = [0] * p
            for v in valid_indices:
                counts[v % p] += 1
            
            min_count = min(counts)
            min_res = counts.index(min_count)
            
            # Filter
            # List comprehension is fast in CPython
            valid_indices = [v for v in valid_indices if v % p != min_res]
            
        if len(valid_indices) >= K:
            best_k = K
            best_set = valid_indices[:K]
            break
            
    return best_set
