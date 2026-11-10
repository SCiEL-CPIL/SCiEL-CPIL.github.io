import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590
    
    def time_left():
        return deadline - time.time()
    
    # Generate norm-4 candidates: 4 non-zero entries of ±1
    # R² = 4, need dot <= 2
    candidates = []
    for comb in combinations(range(d), 4):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[comb[0]] = s1
                        v[comb[1]] = s2
                        v[comb[2]] = s3
                        v[comb[3]] = s4
                        candidates.append(v)
    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Precompute dot product matrix (or use efficient checking)
    # For greedy: select vectors one by one, checking compatibility
    
    best_set = None
    best_k = 0
    
    # Precompute all-pairs dot products for fast checking
    # This is 5304 x 5304 = ~28M entries, manageable
    dots = candidates @ candidates.T
    
    # Greedy with degree-based ordering
    # Degree = number of incompatible pairs (dot > 2)
    incompatible = (dots > 2).astype(np.int32)
    np.fill_diagonal(incompatible, 0)
    degrees = incompatible.sum(axis=1)
    
    # Multiple random restarts with greedy + local search
    rng = np.random.RandomState(42)
    
    for restart in range(50):
        if time_left() < 30:
            break
        
        # Random perturbation of ordering
        if restart == 0:
            order = np.argsort(degrees)  # lowest degree first
        else:
            # Mix: 70% by degree, 30% random
            order = np.arange(n_cand)
            # Shuffle with bias toward low degree
            for i in range(n_cand - 1, 0, -1):
                # Pick from first i+1 with probability biased by degree
                j = rng.randint(0, i + 1)
                if rng.random() < 0.3:
                    order[i], order[j] = order[j], order[i]
        
        selected = []
        selected_set = set()
        
        for idx in order:
            if time_left() < 30:
                break
            # Check if compatible with all selected
            # Use the dot matrix
            if selected:
                sel_indices = np.array(selected)
                if np.any(dots[idx, sel_indices] > 2):
                    continue
            
            selected.append(idx)
            selected_set.add(idx)
        
        k = len(selected)
        
        # Local search: try to swap out one selected for one unselected
        if k > best_k:
            best_k = k
            best_set = list(selected)
        
        # Local improvement: try adding more or swapping
        if time_left() < 30:
            break
        
        improved = True
        while improved and time_left() > 30:
            improved = False
            selected_arr = np.array(selected)
            unselected = [i for i in range(n_cand) if i not in selected_set]
            
            for out_idx in list(selected):
                if time_left() < 30:
                    break
                # Try removing out_idx and adding an unselected one
                remaining = [s for s in selected if s != out_idx]
                remaining_arr = np.array(remaining) if remaining else np.array([], dtype=int)
                
                for in_idx in unselected:
                    if time_left() < 30:
                        break
                    # Check compatibility with remaining
                    if remaining:
                        if np.any(dots[in_idx, remaining_arr] > 2):
                            continue
                    # Also check in_idx != out_idx (already ensured)
                    # Swap
                    selected = remaining + [in_idx]
                    selected_set.discard(out_idx)
                    selected_set.add(in_idx)
                    improved = True
                    unselected.remove(in_idx)
                    unselected.append(out_idx)
                    break
        
        k = len(selected)
        if k > best_k:
            best_k = k
            best_set = list(selected)
    
    # Build result
    if best_set is not None:
        points = candidates[best_set]
    else:
        # Fallback: norm-2
        r2_points = []
        for i in range(d):
            for j in range(i+1, d):
                for s_i in [1, -1]:
                    for s_j in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s_i
                        v[j] = s_j
                        r2_points.append(v)
        points = np.array(r2_points)
        best_k = len(points)
    
    points = np.around(points).astype(np.int64)
    return points
