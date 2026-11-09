import time
import random
import numpy as np
from itertools import combinations

N = 192
MAX_K = 3
MAX_SIZE = 48
MAX_SUM = 3 * N  # 576

def _precompute():
    """Precompute multiples for fast divisibility checks."""
    # divs[a] = set of multiples of a up to MAX_SUM
    # Actually, for vectorization, it's better to just use modulo.
    # But we can precompute a lookup table for "does a divide s?"
    # Let's stick to modulo for now, it's fast enough in vectorized numpy.
    pass

def _can_add_vectorized(x, selected_arr):
    """
    Check if x can be added to selected_arr (numpy array of ints).
    Returns True if valid, False otherwise.
    Conditions:
    1. x does not divide sum of any 3 distinct others in selected_arr.
    2. No a in selected_arr divides sum of x and any 2 distinct others in selected_arr.
    """
    m = len(selected_arr)
    if m == 0:
        return True
    
    # Condition 1: x divides sum of 3 from selected_arr
    if m >= 3:
        # Generate all triplets of indices
        # This is O(m^3) memory/time for indices, but m is small (<50)
        # We can use np.triu_indices or just loops to get sums, then vectorize the mod check.
        # Better: Compute all triplet sums using broadcasting? 
        # For m=50, 50^3 = 125,000 sums. This is fine.
        
        # Efficient way to get all triplet sums:
        # We need sums of 3 distinct elements.
        # Let's just iterate pairs and add the third? No, that's still loops.
        # Let's use a helper to get triplet sums efficiently enough.
        
        # Since m is small, we can just do:
        sums = []
        for i in range(m):
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    sums.append(selected_arr[i] + selected_arr[j] + selected_arr[k])
        
        if sums:
            sums_arr = np.array(sums)
            # Check if any sum is divisible by x
            if np.any(sums_arr % x == 0):
                return False
    
    # Condition 2: For each a in selected_arr, a does not divide (x + b + c) for any b,c in selected_arr \ {a}
    # This is equivalent to: No a in selected_arr divides (x + b + c) where b,c are distinct from a and each other.
    
    # We can check this by iterating over each a in selected_arr.
    # For a fixed a, we need to check if a divides x + b + c for any pair (b,c) from selected_arr \ {a}.
    
    for i, a in enumerate(selected_arr):
        # Get others
        others_mask = np.ones(m, dtype=bool)
        others_mask[i] = False
        others = selected_arr[others_mask]
        
        if len(others) < 2:
            continue
            
        # We need to check if a divides x + b + c for any b,c in others.
        # Let S = x + b + c. We want S % a == 0.
        # This is equivalent to (b + c) % a == (-x) % a.
        
        # Generate all pair sums of others
        # m is small, so O(m^2) is fine.
        pair_sums = []
        for p in range(len(others)):
            for q in range(p + 1, len(others)):
                pair_sums.append(others[p] + others[q])
        
        if pair_sums:
            ps_arr = np.array(pair_sums)
            target = (-x) % a
            if np.any(ps_arr % a == target):
                return False
                
    return True

def _is_valid_vectorized(selected_arr):
    """Verify the entire set is valid."""
    m = len(selected_arr)
    if m < 4:
        return True
        
    for i, a in enumerate(selected_arr):
        others_mask = np.ones(m, dtype=bool)
        others_mask[i] = False
        others = selected_arr[others_mask]
        
        if len(others) < 3:
            continue
            
        # Check if a divides sum of any 3 from others
        triplet_sums = []
        for p in range(len(others)):
            for q in range(p + 1, len(others)):
                for r in range(q + 1, len(others)):
                    triplet_sums.append(others[p] + others[q] + others[r])
                    
        if triplet_sums:
            ts_arr = np.array(triplet_sums)
            if np.any(ts_arr % a == 0):
                return False
    return True

def _greedy_build(order, limit=MAX_SIZE):
    """Greedy: add elements in given order if they don't conflict."""
    selected = []
    for x in order:
        if len(selected) >= limit:
            break
        if _can_add_vectorized(x, np.array(selected)):
            selected.append(x)
    return selected

def _local_search(selected, time_limit):
    """Local search with 1-add and 1-swap moves."""
    start = time.time()
    best = list(selected)
    best_size = len(best)
    best_arr = np.array(best)
    
    while time.time() - start < time_limit:
        improved = False
        
        # 1. Try to add any element
        if len(best) < MAX_SIZE:
            # To speed this up, we can shuffle candidates or just iterate.
            # Iterating 192 candidates * O(S^3) check is expensive if S is large.
            # But vectorized check is fast. Let's try a few random candidates first, then all.
            candidates = [x for x in range(1, N + 1) if x not in best]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if time.time() - start > time_limit:
                    break
                if _can_add_vectorized(x, best_arr):
                    best.append(x)
                    best_arr = np.array(best)
                    improved = True
                    added = True
                    break
            
            if added:
                continue # Restart search with new set
        
        # 2. Try 1-swap (remove one, add one)
        # Only do this if we are stuck or trying to improve diversity?
        # Since we want max cardinality, and add failed, swap might not increase size.
        # However, a swap might allow a future add.
        # Given time constraints, we might skip complex swaps if "Add" is the primary driver.
        # But let's try a limited number of swaps to escape local minima.
        
        if len(best) > 0:
            # Try removing a random element and adding a random candidate
            # This is a random walk.
            # Let's try a few random swaps.
            for _ in range(10): # Limit swaps per iteration to avoid timeout
                if time.time() - start > time_limit:
                    break
                i = random.randint(0, len(best) - 1)
                temp_list = best[:i] + best[i+1:]
                temp_arr = np.array(temp_list)
                
                candidates = [x for x in range(1, N + 1) if x not in temp_list]
                random.shuffle(candidates)
                
                # Try to add a candidate to temp
                swapped = False
                for x in candidates[:50]: # Check only 50 candidates per swap attempt
                    if _can_add_vectorized(x, temp_arr):
                        # Check if this swap is "better" or at least valid.
                        # Since we want max size, and size is same, we accept if it's valid.
                        # To avoid cycling, we might need a memory of visited states, but that's complex.
                        # Simple acceptance: accept if valid.
                        best = temp_list + [x]
                        best_arr = np.array(best)
                        improved = True
                        swapped = True
                        break
                
                if swapped:
                    break
                    
        if not improved:
            # If no add and no successful swap found in this iteration, 
            # we might be stuck. 
            # We can break or continue with random restarts of the local search phase?
            # The outer loop in construct() will handle restarts.
            break
            
    return best

def construct():
    start = time.time()
    # Safety margin: aim to finish by 55s
    deadline = start + 55.0 
    best_result = []
    
    # Strategy 1: Greedy large-first (often works well for divisibility constraints)
    r = _greedy_build(list(range(N, 0, -1)))
    best_result = r
    
    # Strategy 2: Greedy small-first
    r = _greedy_build(list(range(1, N + 1)))
    if len(r) > len(best_result):
        best_result = r
    
    # Strategy 3: Randomized Greedy
    # We will run multiple random greedy builds, and then apply local search to the best ones.
    
    # First, find a good seed set via random greedy
    best_seed = best_result
    best_seed_size = len(best_seed)
    
    num_randoms = 20
    for i in range(num_randoms):
        if time.time() > deadline - 20:
            break
        random.seed(42 + i)
        order = list(range(1, N + 1))
        random.shuffle(order)
        r = _greedy_build(order)
        if len(r) > best_seed_size:
            best_seed = r
            best_seed_size = len(r)
            best_result = r
            
    # Now, apply local search to the best seed found
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 1.0:
            best_result = _local_search(best_result, remaining)
    
    # If we have time, try local search on other promising seeds
    # We saved some random orders? No, we didn't. Let's just do a few more random starts + local search.
    
    for i in range(10):
        if time.time() > deadline - 5:
            break
        random.seed(1000 + i)
        order = list(range(1, N + 1))
        random.shuffle(order)
        r = _greedy_build(order)
        
        # Only do local search if this seed is competitive
        if len(r) >= len(best_result) - 2:
            remaining = min(deadline - time.time() - 2, 2.0)
            if remaining > 0.5:
                r_ls = _local_search(r, remaining)
                if len(r_ls) > len(best_result):
                    best_result = r_ls
    
    # Final polish on the best result
    if time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        if remaining > 0.5:
            best_result = _local_search(best_result, remaining)
            
    # Verify validity
    if not _is_valid_vectorized(np.array(best_result)):
        # Fallback to a simple valid set if verification fails (should not happen)
        best_result = _greedy_build(list(range(N, 0, -1)))
        
    return best_result
