import time
import random
import sys

# LCM of 1..22
_LCM = 232792560
_WEIGHTS = [_LCM // i for i in range(1, 23)]

def _find_collision(indices):
    """
    Returns (is_dissociated, conflict_indices).
    conflict_indices is a set of original indices involved in a collision, or None.
    """
    if not indices:
        return True, None
    
    # Map local index to original index
    # local i corresponds to indices[i]
    # We use a bitmask of local positions to track subset composition
    
    sums = {0: 0} # sum -> mask
    
    for i, orig_idx in enumerate(indices):
        w = _WEIGHTS[orig_idx - 1]
        bit = 1 << i
        # Iterate over a copy of keys because we are modifying the dict
        # Or better, build a new dict or list of new entries
        new_entries = {}
        for s, mask in sums.items():
            ns = s + w
            if ns in sums:
                # Collision found between existing mask for ns and new mask (mask | bit)
                m1 = sums[ns]
                m2 = mask | bit
                diff_mask = m1 ^ m2
                conflict_local = []
                for j in range(len(indices)):
                    if (diff_mask >> j) & 1:
                        conflict_local.append(indices[j])
                return False, set(conflict_local)
            else:
                new_entries[ns] = mask | bit
        sums.update(new_entries)
        
    return True, None

def _construct_remove_strategy(strategy, deadline, seed):
    """
    Start with all elements, remove elements causing collisions.
    strategy: 'min', 'max', 'random'
    """
    random.seed(seed)
    S = set(range(1, 23))
    
    while time.time() < deadline:
        is_diss, conflict = _find_collision(list(S))
        
        if is_diss:
            return list(S)
            
        if not conflict:
            return list(S) # Should not happen if is_diss is False
            
        # Choose element to remove based on strategy
        if strategy == 'min':
            x = min(conflict, key=lambda i: _WEIGHTS[i-1])
        elif strategy == 'max':
            x = max(conflict, key=lambda i: _WEIGHTS[i-1])
        else: # random
            x = random.choice(list(conflict))
            
        S.remove(x)
        
    return list(S)

def _construct_greedy_add(deadline, seed):
    """
    Greedy addition: add elements one by one if they don't cause collision.
    Order elements by some heuristic.
    """
    random.seed(seed)
    S = []
    remaining = list(range(1, 23))
    
    # Sort by weight descending (larger weights first) or random
    # Larger weights are "safer" in terms of magnitude, but let's try random order first
    # Actually, let's try sorting by weight descending.
    remaining.sort(key=lambda i: _WEIGHTS[i-1], reverse=True)
    
    for elem in remaining:
        if time.time() > deadline:
            break
        test_S = S + [elem]
        is_diss, _ = _find_collision(test_S)
        if is_diss:
            S.append(elem)
            
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Removal based
    # Try multiple random seeds and strategies
    strategies = ['min', 'max', 'random']
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for strat in strategies:
        for seed in seeds:
            if time.time() > deadline:
                break
            S = _construct_remove_strategy(strat, deadline, seed)
            if len(S) > best_len:
                best_set = S
                best_len = len(S)
                
    # Strategy 2: Greedy Add
    for seed in range(20):
        if time.time() > deadline:
            break
        S = _construct_greedy_add(deadline, seed)
        if len(S) > best_len:
            best_set = S
            best_len = len(S)
            
    # Strategy 3: Local Search / Hill Climbing on the best set
    # Try to add back removed elements or swap elements
    if best_set:
        best_set_set = set(best_set)
        all_elements = set(range(1, 23))
        removed = all_elements - best_set_set
        
        # Try adding any removed element
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for elem in list(removed):
                test_set = best_set_set | {elem}
                is_diss, _ = _find_collision(list(test_set))
                if is_diss:
                    best_set_set = test_set
                    removed.remove(elem)
                    improved = True
                    break # Restart loop to try adding more
            
        # Try swapping: remove one from best, add one from removed
        # This is more complex, maybe skip if time is tight
        # But let's try a few swaps
        if time.time() < deadline - 5:
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                if not removed:
                    break
                # Pick random elem from best and random from removed
                if not best_set_set:
                    break
                remove_elem = random.choice(list(best_set_set))
                add_elem = random.choice(list(removed))
                test_set = (best_set_set - {remove_elem}) | {add_elem}
                is_diss, _ = _find_collision(list(test_set))
                if is_diss:
                    best_set_set = test_set
                    removed.remove(add_elem)
                    removed.add(remove_elem)
                    
        best_set = list(best_set_set)

    if not best_set:
        # Fallback: return a single element
        return [1]
        
    return best_set
