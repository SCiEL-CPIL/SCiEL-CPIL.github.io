import time
import random
import numpy as np

N = 5000

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

_PRIMES = _primes(N)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    best_size = 0
    best_set = set()
    
    def run_greedy(randomize=False):
        mask = np.ones(N + 1, dtype=bool)
        
        while True:
            if time.time() > deadline:
                return np.where(mask)[0]
            
            points = np.where(mask)[0]
            k = len(points)
            
            if k == 0:
                return points
            
            # Find primes <= k
            # Using list comprehension is fast enough for ~600 primes
            relevant_primes = [p for p in _PRIMES if p <= k]
            
            violated = False
            best_p = None
            best_r = None
            min_count = float('inf')
            
            for p in relevant_primes:
                # Count residues
                counts = np.bincount(points % p, minlength=p)
                min_c = counts.min()
                
                if min_c == 0:
                    continue
                
                violated = True
                if min_c < min_count:
                    min_count = min_c
                    best_p = p
                    if randomize:
                        min_res = np.where(counts == min_c)[0]
                        best_r = min_res[random.randint(0, len(min_res)-1)]
                    else:
                        best_r = counts.argmin()
                elif min_c == min_count:
                    if randomize:
                        if p < best_p:
                            best_p = p
                            min_res = np.where(counts == min_c)[0]
                            best_r = min_res[random.randint(0, len(min_res)-1)]
                    else:
                        if p < best_p:
                            best_p = p
                            best_r = counts.argmin()
            
            if not violated:
                return points
            
            # Remove elements
            to_remove = (points % best_p) == best_r
            mask[points[to_remove]] = False
            
    # Run standard greedy
    result = run_greedy(randomize=False)
    best_set = set(result)
    best_size = len(result)
    
    # Run randomized restarts
    # We can run more restarts if the first one was fast
    # Let's just loop until deadline
    while time.time() < deadline:
        result = run_greedy(randomize=True)
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
            
    return sorted(list(best_set))
