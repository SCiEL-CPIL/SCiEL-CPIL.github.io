import time
import random
from itertools import combinations

_all_masks = []
_inter = None
_total = 0

def _init():
    global _all_masks, _inter, _total
    n = 16
    k = 4
    _all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        _all_masks.append(mask)
    _total = len(_all_masks)
    _inter = [[0] * _total for _ in range(_total)]
    for i in range(_total):
        for j in range(_total):
            _inter[i][j] = _all_masks[i] & _all_masks[j]

def _is_compatible(c_idx, family_indices):
    L = len(family_indices)
    for i in range(L):
        a_idx = family_indices[i]
        ac = _inter[a_idx][c_idx]
        for j in range(i + 1, L):
            b_idx = family_indices[j]
            ab = _inter[a_idx][b_idx]
            bc = _inter[b_idx][c_idx]
            if ab == ac == bc:
                return False
    return True

def _run_greedy(order_indices, max_size=250):
    family_indices = []
    for idx in order_indices:
        if len(family_indices) >= max_size:
            break
        if _is_compatible(idx, family_indices):
            family_indices.append(idx)
    return family_indices

def _local_search(family_indices, deadline, rng):
    best = list(family_indices)
    best_len = len(best)
    total = _total
    
    in_family = set(best)
    not_in_family = [i for i in range(total) if i not in in_family]
    
    removal_candidates = best[:]
    tried = 0
    max_tries = 150
    
    for rem_idx in removal_candidates:
        if tried >= max_tries or time.time() > deadline - 2:
            break
        tried += 1
        
        remaining = [x for x in best if x != rem_idx]
        new_family = remaining[:]
        added = 0
        
        candidates = not_in_family[:]
        rng.shuffle(candidates)
        
        for cand in candidates:
            if cand == rem_idx:
                continue
            if time.time() > deadline - 2:
                break
            if len(new_family) >= len(best) + 1:
                break
            if _is_compatible(cand, new_family):
                new_family.append(cand)
                added += 1
            if added > 0 and len(new_family) > best_len:
                break
        
        if len(new_family) > best_len:
            best = new_family
            best_len = len(best)
            in_family = set(best)
            not_in_family = [i for i in range(total) if i not in in_family]
            removal_candidates = best[:]
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    _init()
    total = _total
    n = 16
    rng = random.Random(12345)
    
    best_family = []
    
    # 1. Standard Lexicographic Order
    order = list(range(total))
    fam = _run_greedy(order)
    if len(fam) > len(best_family):
        best_family = fam[:]
    
    # 2. Reverse Lexicographic Order
    if time.time() < deadline - 5:
        order_rev = list(reversed(range(total)))
        fam = _run_greedy(order_rev)
        if len(fam) > len(best_family):
            best_family = fam[:]
    
    # 3. Randomized greedy with multiple restarts
    for attempt in range(40):
        if time.time() > deadline - 10:
            break
        
        order_rand = list(range(total))
        rng.shuffle(order_rand)
        
        fam = _run_greedy(order_rand)
        if len(fam) > len(best_family):
            best_family = fam[:]
            
        # Local search on promising families
        if len(fam) >= len(best_family) - 2 and time.time() < deadline - 5:
            improved = _local_search(fam, deadline, rng)
            if len(improved) > len(best_family):
                best_family = improved[:]
    
    # 4. Universe permutation-based orderings
    for attempt in range(15):
        if time.time() > deadline - 5:
            break
        
        perm = list(range(n))
        rng.shuffle(perm)
        
        keys = []
        for idx in range(total):
            mask = _all_masks[idx]
            elems = []
            for i in range(n):
                if mask & (1 << i):
                    elems.append(perm[i])
            keys.append((tuple(sorted(elems)), idx))
        
        keys.sort()
        order_perm = [idx for _, idx in keys]
        
        fam = _run_greedy(order_perm)
        if len(fam) > len(best_family):
            best_family = fam[:]
        
        if len(fam) >= len(best_family) - 2 and time.time() < deadline - 3:
            improved = _local_search(fam, deadline, rng)
            if len(improved) > len(best_family):
                best_family = improved[:]
    
    # 5. Final local search on best family
    if time.time() < deadline - 3 and len(best_family) > 0:
        improved = _local_search(best_family, deadline, rng)
        if len(improved) > len(best_family):
            best_family = improved[:]
    
    # Convert best_family indices back to lists of 1-based integers
    result = []
    for idx in best_family:
        mask = _all_masks[idx]
        elements = []
        for i in range(n):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
        
    return result
