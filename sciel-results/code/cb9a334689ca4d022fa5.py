import numpy as np
import time
from itertools import combinations
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Generate all 4-sets
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)
    
    # Precompute conflict bitsets efficiently
    # Two 4-sets conflict if they share >= 3 elements
    conflict = [0] * n_sets
    
    # For each pair, check intersection size
    for i in range(n_sets):
        si = all_4sets[i]
        mask = 0
        for j in range(n_sets):
            if i == j:
                continue
            sj = all_4sets[j]
            # Count common elements
            common = sum(1 for x in si if x in sj)
            if common >= 3:
                mask |= (1 << j)
        conflict[i] = mask
    
    # Bitset-based Greedy with add-if-possible
    def greedy_and_add(order):
        selected = []
        sel_set = set()
        available = (1 << n_sets) - 1
        
        # First pass: greedy in given order
        for idx in order:
            if available & (1 << idx):
                selected.append(idx)
                sel_set.add(idx)
                available &= ~conflict[idx]
                available &= ~(1 << idx)
        
        # Second pass: try to add any remaining valid set
        for j in range(n_sets):
            if j in sel_set:
                continue
            # Check if j can be added
            can_add = True
            for k in selected:
                if conflict[k] & (1 << j):
                    can_add = False
                    break
            if can_add:
                selected.append(j)
                sel_set.add(j)
        
        return selected
    
    # Local search: 1-swap to escape local optima
    def local_search(selection, rng, max_iters=100):
        sel = selection[:]
        sel_set = set(sel)
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try to add any valid set
            for j in range(n_sets):
                if j in sel_set:
                    continue
                can_add = True
                for k in sel:
                    if conflict[k] & (1 << j):
                        can_add = False
                        break
                if can_add:
                    sel.append(j)
                    sel_set.add(j)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 1-swap: remove one, add one different
            if len(sel) > 0:
                i_idx = rng.randint(0, len(sel) - 1)
                i_val = sel[i_idx]
                
                # Remove i_val
                sel.pop(i_idx)
                sel_set.remove(i_val)
                
                # Find a valid j to add (different from i_val)
                forbidden = 0
                for k in sel:
                    forbidden |= conflict[k]
                
                candidates = []
                for j in range(n_sets):
                    if j in sel_set:
                        continue
                    if j == i_val:
                        continue
                    if forbidden & (1 << j):
                        continue
                    candidates.append(j)
                
                if candidates:
                    j_val = rng.choice(candidates)
                    sel.append(j_val)
                    sel_set.add(j_val)
                    improved = True
                else:
                    # Revert
                    sel.append(i_val)
                    sel_set.add(i_val)
        
        return sel
    
    rng = random.Random(42)
    
    best_selection = []
    best_size = 0
    
    # Multi-start greedy with diverse orderings
    n_trials = 200
    for trial in range(n_trials):
        if time.time() > deadline - 120:
            break
        
        # Create diverse ordering: mix of random and structured
        if trial % 3 == 0:
            order = list(range(n_sets))
            rng.shuffle(order)
        elif trial % 3 == 1:
            # Reverse order
            order = list(range(n_sets - 1, -1, -1))
        else:
            # Random with bias toward certain sets
            order = list(range(n_sets))
            rng.shuffle(order)
        
        sel = greedy_and_add(order)
        
        # Apply local search
        sel = local_search(sel, rng, max_iters=50)
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_selection = sel[:]
    
    # Additional local search on best solution
    if time.time() < deadline - 60:
        best_selection = local_search(best_selection, rng, max_iters=200)
        best_size = len(best_selection)
    
    # Generate points from selected 4-sets
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
    
    # Fallback: if no points, use 2-sets
    if not points:
        for i, j in combinations(range(d), 2):
            for si, sj in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
