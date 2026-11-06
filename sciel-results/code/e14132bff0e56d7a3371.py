"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

N = 5000
TIME_LIMIT = 290.0  # safety margin before 300s

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

# Pre-compute indices for each prime and residue
_all_primes_N = _primes(N + 1)
_indices_cache = {}

def _get_indices(p):
    """Get pre-computed indices for prime p, or compute and cache them."""
    if p not in _indices_cache:
        _indices_cache[p] = [np.arange(r, N + 1, p) for r in range(p)]
    return _indices_cache[p]

def _greedy_with_lookahead(primes_order, active_mask, N, lookahead_depth=1):
    """
    Choose residue for each prime using lookahead to minimize total removals.
    For each prime, try the top-K smallest residue classes and evaluate the
    impact on the next prime(s) in the sequence.
    """
    residues = {}
    for i, p in enumerate(primes_order):
        indices_p = _get_indices(p)
        
        # Count active elements in each residue class
        counts = np.zeros(p, dtype=np.int32)
        for r in range(p):
            counts[r] = np.sum(active_mask[indices_p[r]])
        
        # Get sorted residue classes by count (ascending)
        sorted_res = np.argsort(counts)
        
        # Try top K candidates
        K = min(5, len(sorted_res))
        best_r = sorted_res[0]
        best_score = -counts[best_r]  # We want to minimize removals
        
        for k in range(K):
            r = sorted_res[k]
            current_removal = counts[r]
            
            # Lookahead: estimate impact on next prime
            future_cost = 0
            if lookahead_depth > 0 and i + 1 < len(primes_order):
                q = primes_order[i + 1]
                if q <= N + 1:
                    indices_q = _get_indices(q)
                    # Simulate: what would be the min removal for q if we removed class r of p?
                    # Approximate: check if removing class r of p would make q already admissible
                    # Count active in each q-class after removing p-class r
                    # This is expensive, so we approximate by just checking if q is already covered
                    q_counts = np.zeros(q, dtype=np.int32)
                    for rq in range(q):
                        q_counts[rq] = np.sum(active_mask[indices_q[rq]])
                    
                    # If q is currently covered (all q classes have active elements),
                    # removing some elements might help
                    if len(np.where(q_counts > 0)[0]) == q:
                        # Estimate: the min q-class size after removing p-class r
                        # Approximate by just taking min q-count
                        future_cost = np.min(q_counts)
            
            score = -(current_removal + future_cost)
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[p] = int(best_r)
        # Remove all active x with x % p == best_r
        active_mask[indices_p[int(best_r)]] = False
    
    return residues

def _is_admissible_fast(S, K):
    """Check if S is admissible - for every prime p <= len(S), some residue class mod p is missed."""
    K_actual = len(S)
    if K_actual == 0:
        return True
    primes = _primes(K_actual)
    S_arr = np.array(S)
    for p in primes:
        residues_present = np.unique(S_arr % p)
        if len(residues_present) == p:
            return False
    return True

def _local_search_swap(S, removed, K, primes_to_use, deadline):
    """
    Try to improve by swapping residue choices for primes.
    For each prime, try changing its residue to see if we can recover more elements.
    """
    S_set = set(S)
    active_mask = np.ones(N + 1, dtype=bool)
    for x in S:
        active_mask[x] = True
    
    # Current residue assignment
    residues = {}
    for p in primes_to_use:
        # Find which residue class is missing
        indices_p = _get_indices(p)
        missing = None
        for r in range(p):
            if not np.any(active_mask[indices_p[r]]):
                missing = r
                break
        if missing is not None:
            residues[p] = missing
    
    best_S = list(S_set)
    best_size = len(best_S)
    
    # Try swapping residues for each prime
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in primes_to_use:
            if time.time() > deadline:
                break
            current_r = residues.get(p, -1)
            indices_p = _get_indices(p)
            
            for new_r in range(p):
                if new_r == current_r:
                    continue
                
                # Simulate: change residue from current_r to new_r
                # Remove new_r class, add back current_r class
                new_mask = active_mask.copy()
                new_mask[indices_p[new_r]] = False
                if current_r >= 0:
                    new_mask[indices_p[current_r]] = True
                
                new_S = [x for x in range(N + 1) if new_mask[x]]
                if len(new_S) > best_size and len(new_S) <= K:
                    if _is_admissible_fast(new_S, K):
                        active_mask = new_mask
                        residues[p] = new_r
                        best_S = new_S
                        best_size = len(best_S)
                        improved = True
                        break
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 10  # safety margin
    
    # Precompute primes up to N
    all_primes = _primes(N)
    
    best_S = None
    best_size = 0
    
    # Strategy: Try different K values with multiple initializations
    # K determines which primes constrain admissibility
    K_values = [400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]
    
    for K in K_values:
        if time.time() > deadline - 20:
            break
        
        primes_to_use = [p for p in all_primes if p <= K]
        
        # Multiple initialization strategies
        num_trials = 3
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            random.seed(trial * 12345 + K + 7)
            
            active_mask = np.ones(N + 1, dtype=bool)
            
            # Different prime orderings
            primes_order = primes_to_use[:]
            if trial == 0:
                # Smallest first (most constrained first)
                primes_order.sort()
            elif trial == 1:
                # Largest first
                primes_order.sort(reverse=True)
            else:
                # Random shuffle
                random.shuffle(primes_order)
            
            # Use greedy with lookahead
            residues = _greedy_with_lookahead(primes_order, active_mask, N, lookahead_depth=1)
            
            # Compute S
            S = [x for x in range(N + 1) if active_mask[x]]
            size = len(S)
            
            # Check admissibility
            if size <= K and _is_admissible_fast(S, K):
                if size > best_size:
                    best_size = size
                    best_S = S[:]
                
                # Local search: try to swap residues to recover more elements
                if time.time() < deadline - 5:
                    improved_S = _local_search_swap(S, None, K, primes_to_use, deadline - 5)
                    if len(improved_S) > best_size:
                        best_size = len(improved_S)
                        best_S = improved_S[:]
    
    # Fallback: use seed algorithm if nothing found
    if best_S is None:
        points = [1] + [p for p in all_primes if p * p > N]
        changed = True
        while changed:
            changed = False
            for p in _primes(len(points)):
                classes = {}
                for v in points:
                    classes.setdefault(v % p, []).append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    points = [v for v in points if v not in smallest]
                    changed = True
                    break
        best_S = points
    
    return best_S
