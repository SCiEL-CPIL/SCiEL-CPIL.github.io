"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _primes_up_to(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    
    # We'll work with a range up to some max_L for evaluation
    # Start with a reasonable upper bound
    max_eval = 20000
    
    # Greedy construction: for each prime, pick residue that maximizes min coverage
    # in the prefix, preferring to cover areas with low coverage
    residues = [0] * n_primes
    
    # Build coverage array incrementally
    # coverage[x] = number of primes p where x % p == residues[i]
    
    # Greedy: process primes in order, for each choose best residue
    # We track coverage for x in 1..max_eval
    coverage = np.zeros(max_eval + 1, dtype=np.int32)
    
    # For greedy, we want to maximize the minimum coverage in 1..L for some L
    # Strategy: for each prime, try all residues and pick the one that
    # maximizes the length of prefix where min coverage >= 2
    
    # Actually, let's think differently. We want every x in 1..L to have
    # coverage >= 2. So we want to maximize L such that min(coverage[1:L+1]) >= 2.
    
    # Greedy approach: for each prime, choose residue to help extend the doubly-covered prefix.
    # Key insight: if we have a prefix [1, L] where all have coverage >= 2, and we add
    # a new prime, we want to choose its residue so that it helps maintain coverage >= 2
    # for as long as possible.
    
    # Better greedy: For each prime p_i, try each residue r in 0..p_i-1, and compute
    # the resulting "doubly covered prefix length" if we use that residue. Pick the best.
    
    # But this is expensive. Let's use a smarter approach.
    
    # Alternative: Start with all zeros, then do local search.
    
    # Let's do greedy first, then SA.
    
    # Greedy construction:
    # For each prime, choose the residue that maximizes the minimum coverage
    # in the range 1..some_target, where we prioritize extending the prefix.
    
    # I'll use a different greedy: for each prime, choose residue to maximize
    # the number of positions in 1..max_eval that would have coverage exactly 1
    # (i.e., help "complete" pairs) or minimize positions with coverage 0.
    
    # Actually, the cleanest greedy: maximize the length L such that min_{x<=L} coverage[x] >= 2.
    
    def compute_score(res, primes, max_eval):
        """Compute the largest L such that every x in 1..L has coverage >= 2."""
        cov = np.zeros(max_eval + 1, dtype=np.int32)
        for r, p in zip(res, primes):
            # x % p == r means x = r + k*p for k >= 0
            # x in 1..max_eval
            start = r
            if start < 1:
                start += p
            if start > max_eval:
                continue
            cov[start::p] += 1
        
        # Find largest L where min(cov[1:L+1]) >= 2
        # cov[x] >= 2 for all x in 1..L
        min_cov = np.minimum.accumulate(cov[1:max_eval+1])
        # Find last position where min_cov >= 2
        # min_cov is non-decreasing? No, it's the running minimum.
        # Actually, min_cov[i] = min(cov[1], cov[2], ..., cov[i+1])
        # We want the largest i such that min_cov[i] >= 2
        if min_cov[0] < 2:
            return 0
        # Find the last index where min_cov >= 2
        # Since min_cov is non-increasing (running minimum), once it drops below 2, it stays below
        bad = np.where(min_cov < 2)[0]
        if len(bad) == 0:
            return max_eval
        return bad[0]  # first position where min drops below 2, so L = bad[0]
    
    # Greedy construction
    res = [0] * n_primes
    for i, p in enumerate(primes):
        best_r = 0
        best_score = -1
        # Try a subset of residues if p is large, otherwise all
        if p <= 100:
            candidates = range(p)
        else:
            # Sample residues
            candidates = list(range(0, p, max(1, p // 50)))
            candidates.append(p - 1)
        
        for r in candidates:
            res[i] = r
            score = compute_score(res, primes, max_eval)
            if score > best_score:
                best_score = score
                best_r = r
            if time.time() > deadline:
                break
        res[i] = best_r
        if time.time() > deadline:
            break
    
    # Evaluate current best
    current_score = compute_score(res, primes, max_eval)
    
    # Simulated Annealing refinement
    if time.time() < deadline:
        best_res = res[:]
        best_score = current_score
        
        # SA parameters
        T = 5.0
        cooling = 0.9995
        min_T = 0.01
        
        random.seed(42)
        
        while time.time() < deadline and T > min_T:
            # Pick a random prime to change
            i = random.randint(0, n_primes - 1)
            old_r = res[i]
            p = primes[i]
            
            # Choose new residue
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            res[i] = new_r
            new_score = compute_score(res, primes, max_eval)
            
            if new_score > best_score:
                best_res = res[:]
                best_score = new_score
                T = 5.0  # Reset temperature on improvement
            elif new_score >= best_score or random.random() < np.exp((new_score - best_score) / max(T, 0.001)):
                # Accept
                pass
            else:
                # Reject
                res[i] = old_r
            
            T *= cooling
            
            # Periodically check time
            if time.time() > deadline - 5:
                break
        
        # Restore best
        res = best_res[:]
        current_score = best_score
    
    # Final evaluation with a larger range to get accurate score
    # The score is capped at 10^5, but we need to find the actual L
    # Let's evaluate up to min(100000, best_score + 10000) to be safe
    
    final_max = min(100000, current_score + 5000)
    if final_max < 1000:
        final_max = 1000
    
    final_score = compute_score(res, primes, final_max)
    
    return res

# EVOLVE-BLOCK-END
