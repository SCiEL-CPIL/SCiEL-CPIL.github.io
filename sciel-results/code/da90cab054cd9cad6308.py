import numpy as np
import time

def get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = get_primes(1000)
N_PRIMES = len(PRIMES)
SEARCH_LIMIT = 5000

def score_residues(residues, limit):
    counts = np.zeros(limit, dtype=np.int32)
    for p, r in zip(PRIMES, residues):
        if r == 0:
            start = p
        else:
            start = r
        if start > limit:
            continue
        idx = start - 1
        counts[idx:limit:p] += 1
        
    mask = counts < 2
    if not np.any(mask):
        return limit
    first_fail = np.argmax(mask)
    return int(first_fail)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialize with random residues
    best_residues = [np.random.randint(0, p) for p in PRIMES]
    best_score = score_residues(best_residues, SEARCH_LIMIT)
    
    current_residues = best_residues.copy()
    current_score = best_score
    
    while time.time() < deadline:
        i = np.random.randint(0, N_PRIMES)
        old_r = current_residues[i]
        new_r = np.random.randint(0, PRIMES[i])
        
        if new_r == old_r:
            continue
            
        current_residues[i] = new_r
        new_score = score_residues(current_residues, SEARCH_LIMIT)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues.copy()
        else:
            current_residues[i] = old_r

    # Final check with larger limit if we hit the search limit
    if best_score >= SEARCH_LIMIT - 5:
        if time.time() < deadline - 2:
            final_limit = 10000
            counts = np.zeros(final_limit, dtype=np.int32)
            for p, r in zip(PRIMES, best_residues):
                if r == 0:
                    start = p
                else:
                    start = r
                if start > final_limit:
                    continue
                idx = start - 1
                counts[idx:final_limit:p] += 1
            
            mask = counts < 2
            if not np.any(mask):
                final_score = final_limit
            else:
                final_score = int(np.argmax(mask))
                
            if final_score > best_score:
                best_score = final_score

    return best_residues
