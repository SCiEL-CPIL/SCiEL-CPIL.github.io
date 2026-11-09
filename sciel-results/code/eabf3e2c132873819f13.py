"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000

# EVOLVE-BLOCK-START

def _primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
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
    """Construct the largest admissible tuple in [0, N].
    
    An admissible tuple satisfies: for every prime p <= len(tuple),
    some residue class mod p is completely missed by the tuple.
    """
    start_time = time.time()
    TIME_LIMIT = 280  # Safety margin before 300s evaluator kill
    
    # Start with all integers in [0, N]
    active = [True] * (N + 1)
    L = N + 1
    
    # Precompute primes up to N+1
    max_len = N + 1
    all_primes = _primes(max_len)
    
    # Initialize residue counts for all primes <= N+1
    res_counts = {}
    for p in all_primes:
        if p > L:
            break
        counts = [0] * p
        for v in range(N + 1):
            counts[v % p] += 1
        res_counts[p] = counts
        
    # Identify initially violated primes (all classes > 0)
    violated = set()
    for p in all_primes:
        if p > L:
            break
        if min(res_counts[p]) > 0:
            violated.add(p)
            
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT:
            break
        
        # Find the smallest prime p <= L that is still in violated
        target_p = None
        for p in all_primes:
            if p > L:
                break
            if p in violated:
                target_p = p
                break

        if target_p is None:
            break  # Admissible!
            
        p = target_p
        
        # Find the residue class with the minimum count
        counts = res_counts[p]
        min_count = min(counts)
        min_r = counts.index(min_count)
        
        # Remove all points v such that v % p == min_r and active[v]
        to_remove = [v for v in range(N + 1) if active[v] and v % p == min_r]
        
        for v in to_remove:
            active[v] = False
            L -= 1
            
            # Update counts for all primes q <= L
            for q in all_primes:
                if q > L:
                    continue
                r = v % q
                res_counts[q][r] -= 1
                if res_counts[q][r] == 0:
                    violated.discard(q)
                    
        # p is now satisfied
        violated.discard(p)
        
        # Clean up violated set for primes > L
        # (Optional, but keeps set small)
        # for p_v in list(violated):
        #     if p_v > L:
        #         violated.discard(p_v)
        
    points = [v for v in range(N + 1) if active[v]]
    
    return points

# EVOLVE-BLOCK-END
