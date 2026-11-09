import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Generate all 4-sets
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)
    
    # Precompute conflict bitsets
    # conflict[i] is a bitset of sets that conflict with i
    conflict = [0] * n_sets
    # Map set index to bit mask? No, just use Python integers as bitsets.
    # n_sets = 330. Fits in a few machine words, but Python int is arbitrary precision.
    
    for i in range(n_sets):
        si = set(all_4sets[i])
        mask = 0
        for j in range(n_sets):
            if i == j: continue
            sj = set(all_4sets[j])
            if len(si & sj) >= 3:
                mask |= (1 << j)
        conflict[i] = mask

    # Bitset-based Greedy
    def greedy_bitset(order):
        selected = []
        available = (1 << n_sets) - 1
        for idx in order:
            if available & (1 << idx):
                selected.append(idx)
                available &= ~conflict[idx]
                available &= ~(1 << idx)
        return selected

    # Simulated Annealing with Bitsets
    import random
    rng = random.Random(42)
    
    best_selection = []
    best_size = 0
    
    # Initial Greedy
    for trial in range(500):
        if time.time() > deadline - 60:
            break
        order = list(range(n_sets))
        rng.shuffle(order)
        sel = greedy_bitset(order)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selection = sel[:]

    # Local Search: 1-swap and 2-swap
    # Represent selection as a set of indices for O(1) lookups?
    # Or just a list.
    
    def count_conflicts_bitset(selection):
        # Not needed if we maintain validity
        pass

    # We maintain best_selection as a list of valid indices.
    # To check if a new set of indices is valid, we can just check pairwise conflicts.
    # But for speed in search, let's use a helper.
    
    # Optimized 1-swap
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        sel_set = set(best_selection)
        
        # Try 1-swap: remove i, add j
        for i in best_selection:
            # Find neighbors of i that are NOT in sel_set
            # Neighbors of i are in conflict[i]
            # We want j such that j is not in sel_set, and j is not in conflict[k] for any k in sel_set \ {i}
            # Actually, if we remove i, the constraints on j are:
            # 1. j not in sel_set
            # 2. j not in conflict[k] for all k in sel_set \ {i}
            
            # Let's compute the forbidden set for j
            forbidden = 0
            for k in best_selection:
                if k == i: continue
                forbidden |= conflict[k]
            
            # Candidates for j:
            # Not in sel_set
            # Not in forbidden
            # 
            # Let's just iterate j from 0 to n_sets-1
            # This is O(N^2) per swap. 330^2 = 100k. Fast enough.
            
            for j in range(n_sets):
                if j in sel_set:
                    continue
                if forbidden & (1 << j):
                    continue
                # j is a valid replacement
                # New size is same. We want to increase size?
                # 1-swap keeps size same. It's for escaping local minima.
                # To increase size, we need to find a j that allows adding MORE later?
                # Or just do random 1-swaps to explore, then try to add?
                
                # Let's do a different approach:
                # 1. Try to add a new set j to best_selection.
                #    If valid, size increases.
                # 2. If cannot add, try 1-swap to change configuration, then try to add.
                
                pass
            
        # Better Local Search:
        # 1. Try to add any valid set to best_selection.
        # 2. If none, try 1-swap (remove one, add one) to a random valid state, then try to add.
        
        # Step 1: Add
        added = False
        for j in range(n_sets):
            if j in sel_set:
                continue
            # Check if j conflicts with any in best_selection
            valid = True
            for k in best_selection:
                if conflict[k] & (1 << j):
                    valid = False
                    break
            if valid:
                best_selection.append(j)
                sel_set.add(j)
                best_size += 1
                added = True
                improved = True
                break
        
        if added:
            continue
        
        # Step 2: 1-Swap
        # Pick a random i in best_selection, remove it.
        # Pick a random j not in best_selection, add it.
        # Check validity of new set.
        
        if len(best_selection) > 0:
            i_idx = rng.randint(0, len(best_selection)-1)
            i_val = best_selection[i_idx]
            
            # Remove i_val
            best_selection.pop(i_idx)
            sel_set.remove(i_val)
            
            # Find a valid j to add
            # Forbidden by remaining sets
            forbidden = 0
            for k in best_selection:
                forbidden |= conflict[k]
            
            candidates = []
            for j in range(n_sets):
                if j in sel_set:
                    continue
                if forbidden & (1 << j):
                    continue
                candidates.append(j)
            
            if candidates:
                j_val = rng.choice(candidates)
                best_selection.append(j_val)
                sel_set.add(j_val)
                improved = True
            else:
                # Revert
                best_selection.append(i_val)
                sel_set.add(i_val)

    # Generate points
    points = []
    for idx in best_selection:
        s = all_4sets[idx]
        i, j, k, l = s
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)
    
    if not points:
        for i, j in combinations(range(d), 2):
            for si, sj in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
