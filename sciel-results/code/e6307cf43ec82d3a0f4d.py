import time
import random
from itertools import combinations

DEADLINE_SECONDS = 85.0

def _precompute(S):
    """
    Precompute data structures to speed up validity checks for adding a new element to S.
    Returns:
    - three_sums: Set of all sums of 3 distinct elements from S.
    - pair_sums_mod: Dict mapping u in S to a set of (a+b) % u for all distinct a, b in S \ {u}.
    """
    n = len(S)
    three_sums = set()
    pair_sums_mod = {u: set() for u in S}
    
    # Compute three sums
    if n >= 3:
        for c in combinations(S, 3):
            three_sums.add(sum(c))
            
    # Compute pair sums mod u
    # For each u in S, we need sums of pairs from S \ {u}
    for i, u in enumerate(S):
        others = [S[j] for j in range(n) if j != i]
        m = len(others)
        for j in range(m):
            for k in range(j + 1, m):
                s = (others[j] + others[k]) % u
                pair_sums_mod[u].add(s)
                
    return three_sums, pair_sums_mod

def _can_add(v, S, precomp):
    """
    Check if v can be added to S, given precomputed data for S.
    """
    if v in S:
        return False
        
    three_sums, pair_sums_mod = precomp
    
    # 1. Check condition for v:
    # No 3-sum from S is divisible by v
    if v > 0:
        for t in three_sums:
            if t % v == 0:
                return False
                
    # 2. Check condition for each u in S:
    # No 2-sum from S \ {u} plus v is divisible by u
    # i.e., (a + b + v) % u != 0 for all distinct a, b in S \ {u}
    # Equivalent to: (-v) % u not in pair_sums_mod[u]
    for u in S:
        target = (-v) % u
        if target in pair_sums_mod[u]:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Phase 1: Greedy Construction with multiple strategies
    best_set = []
    
    # Strategy 1: Descending order (large numbers first)
    candidates = list(range(192, 0, -1))
    selected = []
    for x in candidates:
        if time.time() > deadline - 5.0:
            break
        # Precompute for current selected set
        if not selected:
            # Empty set is trivially valid
            precomp = (set(), {})
        else:
            precomp = _precompute(selected)
            
        if _can_add(x, selected, precomp):
            selected.append(x)
            
    if len(selected) > len(best_set):
        best_set = list(selected)
        
    # Strategy 2: Randomized Greedy
    rng = random.Random(42)
    for trial in range(20):
        if time.time() > deadline - 10.0:
            break
        candidates = list(range(1, 193))
        rng.shuffle(candidates)
        selected = []
        for x in candidates:
            if time.time() > deadline - 10.0:
                break
            if not selected:
                precomp = (set(), {})
            else:
                precomp = _precompute(selected)
                
            if _can_add(x, selected, precomp):
                selected.append(x)
                
        if len(selected) > len(best_set):
            best_set = list(selected)
            
    # Phase 2: Local Search
    # We try to add elements, and if stuck, try 1-for-2 swaps.
    
    current_set = list(best_set)
    in_set = set(current_set)
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # 1. Try to add any element
        if not current_set:
            precomp = (set(), {})
        else:
            precomp = _precompute(current_set)
            
        added = False
        for x in range(192, 0, -1):
            if x in in_set:
                continue
            if _can_add(x, current_set, precomp):
                current_set.append(x)
                in_set.add(x)
                improved = True
                added = True
                # Restart the addition loop with new precomp
                # Actually, just break and let the while loop re-evaluate
                break
                
        if added:
            continue
            
        # 2. If no element could be added, try 1-for-2 swap
        # Remove one x, add two y, z
        # To limit computation, we only try a few swaps per iteration or randomize
        swap_found = False
        # Shuffle the set to try different removals
        indices = list(range(len(current_set)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 2.0:
                break
                
            x = current_set[idx]
            temp_set = [current_set[j] for j in range(len(current_set)) if j != idx]
            temp_in_set = set(temp_set)
            
            # Precompute for temp_set
            if not temp_set:
                precomp_temp = (set(), {})
            else:
                precomp_temp = _precompute(temp_set)
                
            # Find valid y to add to temp_set
            valid_ys = []
            for y in range(192, 0, -1):
                if y in temp_in_set:
                    continue
                if _can_add(y, temp_set, precomp_temp):
                    valid_ys.append(y)
                    
            # Try to find z for each y
            # To save time, limit the number of y's we try
            if len(valid_ys) > 20:
                valid_ys = valid_ys[:20]
                
            for y in valid_ys:
                if time.time() > deadline - 2.0:
                    break
                    
                new_set_1 = temp_set + [y]
                new_in_set_1 = temp_in_set | {y}
                
                # Precompute for new_set_1
                if not new_set_1:
                    precomp_new1 = (set(), {})
                else:
                    precomp_new1 = _precompute(new_set_1)
                    
                # Find valid z to add to new_set_1
                for z in range(192, 0, -1):
                    if z in new_in_set_1:
                        continue
                    if _can_add(z, new_set_1, precomp_new1):
                        # Success: x removed, y and z added. Size increases by 1.
                        current_set = new_set_1 + [z]
                        in_set = new_in_set_1 | {z}
                        swap_found = True
                        improved = True
                        break
                        
                if swap_found:
                    break
                    
            if swap_found:
                break
                
    return current_set
