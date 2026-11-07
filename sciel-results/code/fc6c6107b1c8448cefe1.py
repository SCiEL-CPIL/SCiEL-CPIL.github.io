"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
import math

NMAX = 256
UNIVERSE = list(range(2, NMAX + 1))

# Precompute primes
def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

PRIMES = [p for p in range(2, NMAX + 1) if _is_prime(p)]

def _can_add(x, S):
    """
    Check if x can be added to set S.
    S is a list of ints.
    """
    n = len(S)
    for i in range(n):
        b = S[i]
        for j in range(i + 1, n):
            c = S[j]
            if (b * c) % x == 0:
                return False
            if (x * c) % b == 0:
                return False
            if (x * b) % c == 0:
                return False
    return True

def _greedy(order, deadline):
    S = []
    S_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add(x, S):
            S.append(x)
            S_set.add(x)
    return S

def _local_search_improve(S, deadline, max_iters=1000, rng_seed=42):
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(rng_seed)
    
    candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        progress = False
        
        # 1. Try to add candidates
        if candidates:
            # Try up to 20 random candidates
            n_try = min(20, len(candidates))
            trial_candidates = rng.sample(candidates, n_try)
            for x in trial_candidates:
                if _can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    candidates.remove(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    progress = True
                    break
            
        if progress:
            continue
            
        # 2. Try swaps
        if not S:
            break
            
        # Try up to 3 different removals
        for _attempt in range(3):
            if time.time() > deadline:
                break
                
            idx_remove = rng.randrange(len(S))
            removed = S[idx_remove]
            
            S_temp = S[:]
            S_temp.pop(idx_remove)
            S_temp_set = S_set - {removed}
            
            if not candidates:
                continue
                
            # Try up to 20 candidates for replacement
            n_try = min(20, len(candidates))
            trial_candidates = rng.sample(candidates, n_try)
            swapped = False
            for x in trial_candidates:
                if _can_add(x, S_temp):
                    S = S_temp + [x]
                    S_set = S_temp_set | {x}
                    if removed not in candidates:
                        candidates.append(removed)
                    if x in candidates:
                        candidates.remove(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    swapped = True
                    progress = True
                    break
            
            if swapped:
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Start with all primes, which is a known valid set of size 54
    best_S = PRIMES[:]
    best_size = len(best_S)

    # Try to improve by adding composites
    # We can try to add a composite if we remove its prime factors
    # But simple addition is blocked if primes are present.
    # So we must do swaps.
    
    # Run local search to see if we can find a larger set (unlikely, but worth a shot)
    # Use a few different seeds
    for seed in [1, 42, 100]:
        if time.time() > deadline - 5:
            break
        S_improved = _local_search_improve(best_S, deadline - 5, max_iters=500, rng_seed=seed)
        if S_improved and len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)
            # If we found a larger set, try to improve it further
            if time.time() < deadline - 2:
                S_improved2 = _local_search_improve(best_S, deadline - 2, max_iters=200, rng_seed=seed+1)
                if S_improved2 and len(S_improved2) > best_size:
                    best_S = S_improved2
                    best_size = len(S_improved2)

    # If we are still at 54, return the primes (or the improved set if it's also 54 but different)
    # The primes are the most "stable" solution.
    
    return [int(x) for x in best_S]
