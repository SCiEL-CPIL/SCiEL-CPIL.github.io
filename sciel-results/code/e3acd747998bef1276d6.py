import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin before 630s kill

    # --- Step 1: Generate a large pool of low-weight integer vectors ---
    # We generate vectors of weight 1, 2, and 3 (squared norms 1, 2, 3).
    # This gives us a rich candidate set from which to select a valid configuration.
    
    candidates = []
    
    # Weight 1: e_i, -e_i
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        candidates.append(v)
        candidates.append(-v)
        
    # Weight 2: e_i +/- e_j
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    # Weight 3: e_i +/- e_j +/- e_k
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates.append(v)
                            
    # Convert to array and remove duplicates/zero vectors
    unique_candidates = []
    seen = set()
    for v in candidates:
        if np.all(v == 0):
            continue
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)
            
    cand_array = np.array(unique_candidates, dtype=np.int64)
    n_cands = len(cand_array)
    
    # Precompute squared norms
    norms_sq = np.sum(cand_array**2, axis=1)
    
    # --- Step 2: Greedy construction with multiple strategies ---
    # We use a greedy approach but with different orderings and restarts.
    # The key insight from the inspirations is that sorting by norm (ascending) 
    # tends to work well because lower-norm points are "easier" to fit.
    
    best_points = None
    best_k = 0
    
    # The known 220-point solution (weight-2 vectors) is a strong baseline
    # Let's first ensure we have this as a fallback
    weight2_points = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    weight2_points.append(v)
    weight2_array = np.array(weight2_points, dtype=np.int64)
    best_points = weight2_array.copy()
    best_k = len(weight2_array)  # 220
    
    # Now try to improve using the full candidate pool
    rng = random.Random(42)
    
    # Precompute dot product matrix for fast distance calculations
    # dist_sq[i,j] = ||p_i - p_j||^2 = ||p_i||^2 + ||p_j||^2 - 2 p_i.p_j
    dots = cand_array @ cand_array.T
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break
        
        # Different ordering strategies
        if restart % 5 == 0:
            # Sort by norm ascending
            order = sorted(range(n_cands), key=lambda i: (norms_sq[i], i))
        elif restart % 5 == 1:
            # Random shuffle
            order = list(range(n_cands))
            rng.shuffle(order)
        elif restart % 5 == 2:
            # Sort by norm descending
            order = sorted(range(n_cands), key=lambda i: (-norms_sq[i], i))
        elif restart % 5 == 3:
            # Random with slight preference for low norm
            order = list(range(n_cands))
            for i in range(n_cands - 1, 0, -1):
                # Bias toward lower indices (lower norm)
                j = rng.randint(0, i)
                if rng.random() < 0.7:
                    order[i], order[j] = order[j], order[i]
        else:
            # Start with weight-2 points, then add others
            order = list(range(n_cands))
            # Move weight-2 points to front
            w2_indices = [i for i in range(n_cands) if norms_sq[i] == 2]
            others = [i for i in range(n_cands) if norms_sq[i] != 2]
            rng.shuffle(others)
            order = w2_indices + others
        
        selected = []
        selected_indices = []
        selected_norms = []
        min_dist_sq = float('inf')
        
        for idx in order:
            if time.time() > deadline - 10:
                break
            
            p = cand_array[idx]
            pn = norms_sq[idx]
            
            # Check against all selected points
            compatible = True
            current_max_norm = max(selected_norms) if selected else 0
            
            # The condition is: min_dist_sq >= max(max_norm, pn)
            # So for each pair, we need dist_sq >= max(current_max_norm, pn)
            required_threshold = max(current_max_norm, pn)
            
            for si in selected_indices:
                # dist_sq = ||p - p_si||^2 = pn + norms_sq[si] - 2 * dots[idx, si]
                d_sq = pn + norms_sq[si] - 2 * dots[idx, si]
                
                if d_sq < required_threshold:
                    compatible = False
                    break
                
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq
            
            if compatible:
                selected.append(p)
                selected_indices.append(idx)
                selected_norms.append(pn)
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_points = np.array(selected, dtype=np.int64)
            
            # If we found a good solution, try to extend it further
            if best_k > 220 and time.time() < deadline - 60:
                # Try adding more points to the current best
                remaining = [i for i in range(n_cands) if i not in set(selected_indices)]
                rng.shuffle(remaining)
                
                for idx in remaining:
                    if time.time() > deadline - 10:
                        break
                    
                    p = cand_array[idx]
                    pn = norms_sq[idx]
                    
                    compatible = True
                    current_max_norm = max(selected_norms) if selected else 0
                    required_threshold = max(current_max_norm, pn)
                    
                    for si in selected_indices:
                        d_sq = pn + norms_sq[si] - 2 * dots[idx, si]
                        if d_sq < required_threshold:
                            compatible = False
                            break
                        if d_sq < min_dist_sq:
                            min_dist_sq = d_sq
                    
                    if compatible:
                        selected.append(p)
                        selected_indices.append(idx)
                        selected_norms.append(pn)
                
                if len(selected) > best_k:
                    best_k = len(selected)
                    best_points = np.array(selected, dtype=np.int64)
    
    # Final validation and return
    if best_points is None or best_k < 2:
        # Fallback to the 220-point solution
        best_points = weight2_array
        best_k = 220
    
    best_points = np.array(best_points, dtype=np.int64)
    best_points = np.around(best_points).astype(np.int64)
    
    return best_points
