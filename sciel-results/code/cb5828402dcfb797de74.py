"""Rewritten program for the suite task; stronger local-search algorithm with conflict-degree pruning."""
# EVOLVE-BLOCK-START
import time
from random import Random

def _instance(size, seed, limit=10000):
    rng = Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x != 0:
            values.add(x)
    return sorted(values)

def _is_sum_free(S):
    s = set(S)
    for a in S:
        for b in S:
            if (a + b) in s:
                return False
    return True

def _conflicts(S_set, x):
    """Return set of members in S_set that conflict with x (i.e., adding x would violate sum-freeness)."""
    conf = set()
    if 2 * x in S_set:
        conf.add(x)
    for c in S_set:
        if x + c in S_set:
            conf.add(x)
            conf.add(c)
    # c1 + c2 = x
    for c1 in S_set:
        for c2 in S_set:
            if c1 + c2 == x:
                conf.add(c1)
                conf.add(c2)
                conf.add(x)
    return conf

def _can_add(S_set, x):
    """Check if x can be added to S_set without violating sum-freeness."""
    if 2 * x in S_set:
        return False
    # Check x + c in S
    for c in S_set:
        if x + c in S_set:
            return False
    # Check x = c1 + c2
    for c1 in S_set:
        if x - c1 in S_set:
            return False
    return True

def _greedy_by_key(S, key_func):
    """Greedy construction: add elements in order of key_func if they don't conflict."""
    S_set = set()
    result = []
    for x in sorted(S, key=key_func):
        if _can_add(S_set, x):
            S_set.add(x)
            result.append(x)
    return result

def _greedy_odd(S):
    return [x for x in S if x % 2 != 0]

def _prune_by_conflict_degree(S, deadline):
    """Iteratively remove elements with highest conflict out-degree."""
    S_set = set(S)
    while time.time() < deadline and len(S_set) > 0:
        # Compute conflict degree for each element
        degrees = {}
        for x in S_set:
            deg = 0
            for y in S_set:
                if y != x and (x + y) in S_set:
                    deg += 1
            degrees[x] = deg
        
        # Find element with max degree
        if not degrees:
            break
        max_deg = max(degrees.values())
        if max_deg == 0:
            break
        
        # Remove one element with max degree
        to_remove = [x for x in S_set if degrees[x] == max_deg]
        S_set.discard(to_remove[0])
    
    return list(S_set)

def _local_search(S, rng, deadline):
    """Improve a sum-free subset of S via local search with efficient conflict checking."""
    S_set = set(S)
    all_elements = set(S)
    excluded = list(all_elements - S_set)
    
    best = len(S_set)
    best_set = set(S_set)
    
    # Ensure initial is valid
    if not _is_sum_free(S_set):
        S_set = set(_greedy_by_key(list(S), lambda v: -abs(v)))
        excluded = list(all_elements - S_set)
        best = len(S_set)
        best_set = set(S_set)
        
    while time.time() < deadline:
        improved = False
        
        # 1. Try adding any excluded element that fits
        for x in list(excluded):
            if _can_add(S_set, x):
                S_set.add(x)
                excluded.remove(x)
                improved = True
                if len(S_set) > best:
                    best = len(S_set)
                    best_set = set(S_set)
        
        # 2. Try swaps: remove c from S, add x from excluded
        # To speed up, we only try a random subset of swaps if many
        swap_candidates = list(excluded)
        rng.shuffle(swap_candidates)
        for x in swap_candidates:
            if time.time() >= deadline:
                break
            if _can_add(S_set, x):
                continue # Already handled or became addable? No, if it was addable we would have added it.
            # Find conflicts for x
            conflicts = set()
            if 2 * x in S_set:
                conflicts.add(2 * x)
            for c in S_set:
                if x + c in S_set:
                    conflicts.add(c)
                    conflicts.add(x + c)
                if x - c in S_set:
                    conflicts.add(c)
                    conflicts.add(x - c)
            
            for c in conflicts:
                if c in S_set:
                    # Try swapping c with x
                    if _can_add(S_set - {c}, x):
                        S_set.discard(c)
                        S_set.add(x)
                        excluded.remove(x)
                        improved = True
                        if len(S_set) > best:
                            best = len(S_set)
                            best_set = set(S_set)
                        break
        
        if not improved:
            # Stuck. Try a random restart of the local search process?
            # Or just break and let the outer loop handle restarts.
            # For now, break to save time for other restarts.
            break
            
    return list(best_set)

def construct():
    deadline = time.time() + 145.0
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for idx, (size, seed) in enumerate(instances):
        S = _instance(size, seed)
        rng = Random(seed + 12345)
        
        remaining_time = deadline - time.time()
        num_remaining = len(instances) - idx
        inst_budget = remaining_time / num_remaining - 2.0
        inst_deadline = time.time() + inst_budget
        
        # Build multiple initial candidates
        cand1 = _greedy_odd(S)
        cand2 = _greedy_by_key(S, lambda v: -abs(v))
        cand3 = _greedy_by_key(S, lambda v: v)
        cand4 = _greedy_by_key(S, lambda v: -v)
        
        # Conflict-degree pruning candidate
        prune_deadline = min(time.time() + 0.5, inst_deadline)
        pruned = _prune_by_conflict_degree(S, prune_deadline)
        cand5 = _greedy_by_key(pruned, lambda v: -abs(v))
        
        candidates = [cand1, cand2, cand3, cand4, cand5]
        
        best_result = []
        
        # Run local search on the best initial candidate
        best_init = max(candidates, key=len)
        result = _local_search(best_init, rng, inst_deadline)
        if len(result) > len(best_result):
            best_result = result
            
        # If time permits, try local search on other promising initial candidates
        if time.time() < inst_deadline - 5.0:
            for cand in candidates:
                if cand == best_init:
                    continue
                if time.time() >= inst_deadline:
                    break
                sub_deadline = min(time.time() + 2.0, inst_deadline)
                res = _local_search(cand, rng, sub_deadline)
                if len(res) > len(best_result):
                    best_result = res
                    
        # Fallback to best initial if no result
        if not best_result:
            best_result = best_init
            
        if not _is_sum_free(best_result):
            # Safety check
            for cand in candidates:
                if _is_sum_free(cand):
                    best_result = cand
                    break
            if not _is_sum_free(best_result):
                best_result = [x for x in S if x > 0] # Positive numbers are sum-free? No.
                # Actually, if all else fails, take the largest magnitude elements
                best_result = sorted(S, key=abs, reverse=True)[:len(S)//2]
                # Ensure it's sum-free
                while not _is_sum_free(best_result):
                    # Remove one
                    best_result.pop()
        
        results.append(best_result)
    
    return results
# EVOLVE-BLOCK-END
