import numpy as np
import itertools
import time
import random

D = 11
DEADLINE = 145.0

# Pre-compute candidates and conflict matrix
def _init():
    # Generate W4 vectors
    vecs = []
    for pos in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            v[list(pos)] = signs
            vecs.append(v)
    candidates = np.array(vecs)
    n = len(candidates)
    
    # Dot product
    dots = candidates @ candidates.T
    
    # Conflict: dot > 2
    # Since integers, dot > 2 means dot >= 3
    conflict = (dots >= 3).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    return candidates, conflict, n

_CANDIDATES, _CONFLICT, _N = _init()

def _greedy_fill(conflict, n, available_mask, rng):
    """
    Greedy fill: repeatedly pick vertex with min degree in available subgraph.
    available_mask: boolean array of size n, 1 if available.
    """
    selected = []
    remaining = available_mask.copy()
    
    # To speed up, we can precompute degrees? No, degrees change as we remove neighbors.
    # But we can just compute degree in the remaining set each time.
    # deg[i] = sum(conflict[i, remaining])
    
    while remaining.any():
        # Compute degrees
        deg = np.sum(conflict[:, remaining], axis=1)
        deg[~remaining] = n + 1
        
        min_deg = deg.min()
        if min_deg >= n:
            break
            
        candidates_mask = (deg == min_deg) & remaining
        cand_indices = np.where(candidates_mask)[0]
        
        if len(cand_indices) == 0:
            break
            
        # Pick random among min degree
        idx = cand_indices[rng.randint(0, len(cand_indices) - 1)]
        
        selected.append(idx)
        remaining[idx] = False
        # Remove neighbors
        neighbors = np.where(conflict[idx] > 0)[0]
        remaining[neighbors] = False
        
    return selected

def _local_search(conflict, n, selected, rng, time_limit):
    """
    Local search with Add and 1-Swap (to enable future Adds).
    """
    start = time.time()
    sel_set = set(selected)
    sel_list = list(selected)
    
    # Helper to compute addable mask
    def get_addable_mask(S_list):
        mask = np.ones(n, dtype=bool)
        for s in S_list:
            mask &= (conflict[s] == 0)
        # Exclude selected
        mask[list(S_list)] = False
        return mask

    # Initial addable
    addable_mask = get_addable_mask(sel_list)
    
    while time.time() < time_limit:
        improved = False
        
        # Phase 1: Add
        addable_indices = np.where(addable_mask)[0]
        if len(addable_indices) > 0:
            # Pick one with min degree in full graph (less constraining)
            # Or random. Random is faster. Min degree might be better.
            # Let's use min degree in the conflict graph restricted to addable?
            # No, just pick random for speed, or min global degree.
            # Global degree is precomputed?
            # Let's just pick random.
            idx = addable_indices[rng.randint(0, len(addable_indices) - 1)]
            
            sel_list.append(idx)
            sel_set.add(idx)
            
            # Update addable mask
            addable_mask[idx] = False
            addable_mask &= (conflict[idx] == 0)
            
            improved = True
            continue
        
        # Phase 2: 1-Swap
        # Try to swap a random element to see if it opens up addable slots
        if len(sel_list) == 0:
            break
            
        # Sample a few candidates for swap to save time
        num_tries = min(10, len(sel_list))
        for _ in range(num_tries):
            if time.time() > time_limit:
                break
                
            u_idx = rng.randint(0, len(sel_list) - 1)
            u = sel_list[u_idx]
            
            # Temporarily remove u
            temp_list = [x for i, x in enumerate(sel_list) if i != u_idx]
            
            # Compute addable for temp_list
            temp_mask = get_addable_mask(temp_list)
            temp_indices = np.where(temp_mask)[0]
            
            if len(temp_indices) > 0:
                # We can swap u for some v in temp_indices
                # Pick a random v
                v = temp_indices[rng.randint(0, len(temp_indices) - 1)]
                
                # Perform swap
                sel_list[u_idx] = v
                sel_set.remove(u)
                sel_set.add(v)
                
                # Recompute addable mask from scratch (fast enough)
                addable_mask = get_addable_mask(sel_list)
                improved = True
                break # Restart loop with new state
                
    return sel_list

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_set = []
    best_count = 0
    
    # Multiple restarts
    while time.time() < DEADLINE:
        # 1. Greedy Initialization
        # Use a random subset or full set? Full set.
        # To diversify, we can randomly remove some initial candidates?
        # Or just use different RNG seeds for greedy tie-breaking.
        
        init_set = _greedy_fill(_CONFLICT, _N, np.ones(_N, dtype=bool), rng)
        
        # 2. Local Search
        improved_set = _local_search(_CONFLICT, _N, init_set, rng, time_limit=2.0)
        
        if len(improved_set) > best_count:
            best_count = len(improved_set)
            best_set = improved_set
            
        # Check time
        if time.time() > DEADLINE - 10:
            break

    # Build result
    if best_set:
        result = _CANDIDATES[best_set]
    else:
        # Fallback
        result = _CANDIDATES[:10]
        
    return result.astype(np.int64)
