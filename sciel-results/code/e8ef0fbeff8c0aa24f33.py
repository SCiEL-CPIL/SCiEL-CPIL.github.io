"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

# EVOLVE-BLOCK-START

def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

# Precompute primes up to N
_all_primes = _primes(N)

def _greedy_init(k):
    """
    Greedy initialization of residues for primes <= k.
    Returns a list of residues (one per prime <= k) and the final alive array.
    """
    primes_to_use = [p for p in _all_primes if p <= k]
    if not primes_to_use:
        return [], np.ones(N + 1, dtype=bool)
    
    alive = np.ones(N + 1, dtype=bool)
    indices = np.arange(N + 1)
    residues = []
    
    for p in primes_to_use:
        residues_arr = indices % p
        counts = np.bincount(residues_arr, weights=alive, minlength=p)
        min_count = np.min(counts)
        # To break ties deterministically but with some variation, 
        # we can pick the first min, or random among mins.
        # Let's pick the first for now in initialization.
        candidate_residues = np.where(counts == min_count)[0]
        r_min = candidate_residues[0]
        residues.append(r_min)
        
        mask = (residues_arr == r_min)
        alive[mask] = False
        
        if not np.any(alive):
            break
            
    return residues, alive

def _eval_survivors(residues, primes_to_use):
    """
    Evaluate the number of survivors given a set of residues and primes.
    residues is a list of length len(primes_to_use).
    """
    if not residues:
        return N + 1
        
    alive = np.ones(N + 1, dtype=bool)
    indices = np.arange(N + 1)
    
    for p, r in zip(primes_to_use, residues):
        if r < 0 or r >= p:
            # Invalid residue, shouldn't happen
            return 0
        mask = (indices % p) == r
        alive[mask] = False
        if not np.any(alive):
            return 0
            
    return int(np.sum(alive))

def _local_search(residues, primes_to_use, k, max_iters=500):
    """
    Improve residues using local search (1-opt).
    Tries changing one residue at a time to maximize survivors.
    """
    current_residues = residues.copy()
    current_score = _eval_survivors(current_residues, primes_to_use)
    
    improved = True
    while improved:
        improved = False
        for i, p in enumerate(primes_to_use):
            old_r = current_residues[i]
            best_r = old_r
            best_score = current_score
            
            # Try all other residues for this prime
            # Optimization: only try residues that might improve things?
            # For now, try all. p is small enough.
            for new_r in range(p):
                if new_r == old_r:
                    continue
                current_residues[i] = new_r
                score = _eval_survivors(current_residues, primes_to_use)
                if score > best_score:
                    best_score = score
                    best_r = new_r
                # Revert immediately
                current_residues[i] = old_r
                
            if best_r != old_r:
                current_residues[i] = best_r
                current_score = best_score
                improved = True
                
                # Check if we can satisfy the size requirement
                if current_score >= k:
                    # We could stop early, but maybe we can go higher?
                    # Actually, if we are doing local search to maximize survivors,
                    # we should keep going until no single move improves.
                    pass
                    
    return current_residues, current_score

def construct():
    """
    Constructs the largest admissible tuple in [0, N].
    
    Strategy:
    1. Estimate the maximum possible k.
    2. For a candidate k, use greedy initialization to get residues.
    3. Refine residues using local search to maximize survivors.
    4. Binary search for the largest k such that max_survivors >= k.
    5. Construct the set.
    """
    
    # Precomputation
    indices = np.arange(N + 1)
    
    # Binary search for maximum k
    # Lower bound: 1 (or 0)
    # Upper bound: N + 1
    # We know that for small k, it's easy.
    # Let's start with a reasonable guess.
    # The density of survivors for primes up to k is prod(1-1/p).
    # For k=100, primes up to 100. Product is roughly e^-gamma / ln(100) ~ 0.56/4.6 ~ 0.12? No.
    # Mertens theorem: prod_{p<=y} (1-1/p) ~ e^-gamma / ln(y).
    # For y=100, ln(100)~4.6, e^-gamma~0.561. Ratio ~ 0.12.
    # So survivors ~ 0.12 * 5000 = 600.
    # But we are free to choose residues to maximize survivors.
    # The maximum density is higher because we align holes.
    # However, the upper bound on the size is N+1.
    
    # Let's do a binary search on k.
    # f(k) = max_survivors(k) - k. We want f(k) >= 0.
    # f(k) is likely decreasing?
    # As k increases, we add more primes, which generally reduces survivors.
    # Also, the requirement k increases.
    # So f(k) is decreasing.
    
    low = 1
    high = N + 1
    best_k = 1
    best_residues = None
    best_survivors = 0
    best_alive = None
    
    # Optimization: We can skip the binary search and just iterate?
    # Or use the greedy estimate to set a tighter bound.
    
    # Let's run a quick greedy estimate for a few k to get a sense.
    # But binary search is safe.
    
    while low <= high:
        if time.time() > _DEADLINE - 10.0:
            break
            
        mid = (low + high) // 2
        
        # Get primes <= mid
        primes_to_use = [p for p in _all_primes if p <= mid]
        
        # Greedy Init
        residues, alive_greedy = _greedy_init(mid)
        survivors_greedy = int(np.sum(alive_greedy))
        
        # If greedy already satisfies mid, we can try to improve or just accept.
        # If greedy fails, local search might save it?
        # Or if greedy succeeds, local search might allow a larger k?
        
        # Let's always try to improve the greedy result with local search if time permits.
        # But local search is expensive.
        # Only do local search if mid is a "candidate" for the maximum.
        # In binary search, we are checking feasibility.
        # If survivors_greedy >= mid, then mid is feasible.
        # We want to find the MAXIMUM feasible mid.
        
        if survivors_greedy >= mid:
            # Feasible. Try larger.
            best_k = mid
            best_residues = residues
            best_survivors = survivors_greedy
            best_alive = alive_greedy
            low = mid + 1
        else:
            # Not feasible with greedy.
            # Could local search make it feasible?
            # If mid is close to the boundary, maybe.
            # But for simplicity and speed, let's assume greedy is good enough for the boundary.
            # If we want to be robust, we could run a limited local search here.
            
            # Let's try a limited local search if we have time and mid is "interesting"
            # (e.g., close to current best or high).
            # Actually, if greedy says no, and we assume monotonicity, we should go lower.
            # But if greedy is suboptimal, we might miss a feasible k.
            
            # Let's run a quick local search with a small budget to see if it can be fixed.
            # This is a heuristic improvement.
            
            if mid > best_k: # Only if it's potentially better
                # Run local search with limited iterations
                # Note: _local_search is expensive (O(P * p * N))
                # Let's limit it to 10 iterations or so.
                # Actually, _local_search as defined loops until no improvement.
                # Let's create a bounded version or just skip it for now to ensure speed.
                
                # For now, skip local search in the binary search loop to save time.
                # We will do a final local search on the best_k found.
                pass
            
            high = mid - 1

    # Now we have best_k from greedy binary search.
    # Let's try to improve best_k using local search on the residues for best_k.
    # Or even try best_k + 1, best_k + 2, etc. with local search?
    
    # Let's refine the residues for best_k first.
    primes_to_use = [p for p in _all_primes if p <= best_k]
    
    if best_residues is None:
        # Should not happen if best_k >= 1
        residues, best_alive = _greedy_init(best_k)
        best_residues = residues
    else:
        residues = best_residues
        
    # Run local search to maximize survivors for this k
    # We want to see if we can increase survivors, which might allow us to pick a subset of size k
    # (we already have >= k survivors from greedy).
    # Actually, the goal is to find the largest k such that there exists a set of size k.
    # If we have S survivors with S >= k, we can pick k of them.
    # So we just need to find the max k where max_survivors(k) >= k.
    
    # Let's try to push k higher.
    # Current best_k is such that greedy(k) >= k.
    # What about k+1? Greedy(k+1) < k+1.
    # But maybe local_search(k+1) >= k+1?
    
    # Let's try a few values above best_k.
    # We have time budget.
    
    current_best_k = best_k
    current_best_residues = best_residues
    current_best_survivors = best_survivors
    
    # Try k = current_best_k + 1, +2, etc.
    # Limit to a few steps to avoid wasting time.
    # Also, we should refine the current best_k with local search to ensure we have the best residues.
    
    # Refine current best_k
    refined_residues, refined_survivors = _local_search(residues, primes_to_use, best_k, max_iters=10)
    if refined_survivors > current_best_survivors:
        current_best_survivors = refined_survivors
        current_best_residues = refined_residues
        # Update best_alive
        alive = np.ones(N + 1, dtype=bool)
        for p, r in zip(primes_to_use, current_best_residues):
            alive[(indices % p) == r] = False
        best_alive = alive
        
    # Now try to increase k
    # We can try k = current_best_k + 1, 2, 3...
    # For each, we need to find residues.
    # We can initialize with the previous residues (extended by greedy for new primes)
    
    # Let's try up to 5 steps higher.
    for delta_k in range(1, 6):
        new_k = current_best_k + delta_k
        if new_k > N + 1:
            break
        if time.time() > _DEADLINE - 20.0:
            break
            
        new_primes_to_use = [p for p in _all_primes if p <= new_k]
        
        # Initialize residues
        # Start with current_best_residues and add greedy for new primes
        new_residues = current_best_residues.copy()
        
        # Identify new primes
        old_prime_set = set(primes_to_use)
        new_primes = [p for p in new_primes_to_use if p not in old_prime_set]
        
        # We need to re-run greedy or extend?
        # Extending is tricky because the optimal residue for old primes might change
        # given the new constraints.
        # Let's just re-run greedy for the full new_k.
        init_residues, init_alive = _greedy_init(new_k)
        init_survivors = int(np.sum(init_alive))
        
        if init_survivors >= new_k:
            # Feasible with greedy!
            # Update best
            current_best_k = new_k
            current_best_residues = init_residues
            current_best_survivors = init_survivors
            best_alive = init_alive
            # Continue trying higher?
            # Since we are iterating delta_k, we will try new_k+1 next.
            # But we should probably break and restart the loop or just continue.
            # The loop continues to delta_k+1, which is new_k+1.
            # So we naturally try higher.
            continue
        else:
            # Not feasible with greedy.
            # Try local search?
            # This is expensive.
            # Only try if we have lots of time.
            if time.time() < _DEADLINE - 50.0:
                ls_residues, ls_survivors = _local_search(init_residues, new_primes_to_use, new_k, max_iters=20)
                if ls_survivors >= new_k:
                    current_best_k = new_k
                    current_best_residues = ls_residues
                    current_best_survivors = ls_survivors
                    # Construct alive
                    alive = np.ones(N + 1, dtype=bool)
                    for p, r in zip(new_primes_to_use, ls_residues):
                        alive[(indices % p) == r] = False
                    best_alive = alive
                    continue
            # If local search also fails, stop trying higher k.
            # Assume monotonicity failure.
            break

    # Final construction
    if current_best_k == 0:
        return []
        
    # Use best_alive or reconstruct from current_best_residues
    # It's safer to reconstruct to ensure consistency
    final_primes_to_use = [p for p in _all_primes if p <= current_best_k]
    alive = np.ones(N + 1, dtype=bool)
    for p, r in zip(final_primes_to_use, current_best_residues):
        alive[(indices % p) == r] = False
        
    result_indices = np.where(alive)[0]
    
    if len(result_indices) < current_best_k:
        # Should not happen, but fallback
        # Try to reduce k?
        # Or just return what we have?
        # The task requires a valid tuple.
        # If we can't form a tuple of size current_best_k, we must return a smaller one.
        # But we verified survivors >= current_best_k.
        # So this branch is unlikely.
        # Let's just take what we have.
        pass
    
    # Pick exactly current_best_k elements
    # We can pick the first current_best_k
    final_set = result_indices[:current_best_k].tolist()
    
    return final_set

# EVOLVE-BLOCK-END
