"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
from collections import defaultdict


def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Generate candidate vectors
    # We use a structured generation to ensure diversity and manageability.
    # We focus on vectors with small integer coordinates to keep norms low,
    # which allows for higher packing density.
    
    candidates = []
    seen = set()
    
    # Helper to add a vector if not seen
    def add_candidate(v):
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            candidates.append(v)

    # 1. Single component vectors (axes)
    # Norms: 1, 2, 3
    for i in range(D):
        for val in [1, -1, 2, -2, 3, -3]:
            v = [0] * D
            v[i] = val
            add_candidate(v)
            
    # 2. Two component vectors
    # Combinations of indices
    for i in range(D):
        for j in range(i + 1, D):
            # Values: (1,1), (1,-1), (2,1), (1,2), (2,2) etc.
            # Keep norms small: max norm^2 = 2^2+2^2=8.
            for vi in [1, -1, 2, -2]:
                for vj in [1, -1, 2, -2]:
                    v = [0] * D
                    v[i] = vi
                    v[j] = vj
                    add_candidate(v)
                    
    # 3. Three component vectors
    # This increases diversity significantly.
    # Limit to small values to control norm.
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                # Only use 1/-1 to keep norm^2 = 3
                for vi in [1, -1]:
                    for vj in [1, -1]:
                        for vk in [1, -1]:
                            v = [0] * D
                            v[i] = vi
                            v[j] = vj
                            v[k] = vk
                            add_candidate(v)

    if not candidates:
        return np.array([], dtype=np.int64).reshape(0, D)

    cand_arr = np.array(candidates, dtype=np.int64)
    n_cand = len(cand_arr)
    
    # Compute norms squared
    norms_sq = np.sum(cand_arr ** 2, axis=1)
    
    # Strategy:
    # The constraint is min_pairwise_dist >= max_norm.
    # Let R = max_norm of the selected set.
    # We want to maximize |S| subject to ||vi - vj|| >= R for all i,j in S.
    
    # Heuristic: Sort candidates by norm (ascending).
    # Greedily select vectors.
    # When adding a new vector, the "effective" max norm R might increase.
    # However, checking against a fixed global max_norm is a valid sufficient condition 
    # if we ensure the selected set's max norm is <= that global max.
    # Actually, the evaluator checks the *result*. 
    # If we select a set S, and let R_S = max(norm(S)), we need dist >= R_S.
    
    # To be safe and efficient, we can perform the greedy selection 
    # while dynamically tracking the current max norm of the selected set.
    # But checking dist >= current_max_norm is easier if we precompute distances? 
    # No, N is small enough for O(N*K) with NumPy vectorization per step.
    
    # Sort by norm to prefer "small" vectors which allow tighter packing?
    # Or sort by some centrality? 
    # Let's try sorting by norm ascending.
    order = np.argsort(norms_sq)
    
    selected_indices = []
    selected_vecs = []
    
    # We will iterate through candidates in sorted order.
    # For each candidate, check if it is at least `current_max_norm` away from all selected.
    # If yes, add it. Update current_max_norm.
    
    current_max_norm_sq = 0
    
    for idx in order:
        if time.time() > deadline:
            break
        
        v = cand_arr[idx]
        v_norm_sq = norms_sq[idx]
        
        # Potential new max norm
        new_max_norm_sq = max(current_max_norm_sq, v_norm_sq)
        
        # If no vectors selected yet, add first one
        if not selected_vecs:
            selected_indices.append(idx)
            selected_vecs.append(v)
            current_max_norm_sq = v_norm_sq
            continue
        
        # Check distance against all selected vectors
        # We need ||v - u||^2 >= new_max_norm_sq for all u in selected_vecs
        # Note: If v_norm_sq > current_max_norm_sq, the constraint tightens for ALL existing pairs?
        # No, the constraint is pairwise. 
        # If we add v, we need:
        # 1. ||v - u||^2 >= new_max_norm_sq for all u in selected.
        # 2. The existing pairs in selected already satisfied ||u1 - u2||^2 >= current_max_norm_sq.
        #    Since new_max_norm_sq >= current_max_norm_sq, the existing pairs might NOW be invalid 
        #    if new_max_norm_sq > current_max_norm_sq and some existing pair had distance < new_max_norm_sq.
        
        # This dynamic update is dangerous. 
        # Simpler approach: 
        # Pick a target radius R. 
        # Or, just use the global max norm of the *entire candidate pool* as the constraint.
        # This guarantees validity but might be suboptimal if we include huge vectors in the pool.
        # We generated small vectors (max norm^2 = 3^2=9 or 2^2+2^2=8). 
        # Let's use the max norm of the *generated candidates* as the fixed constraint R^2.
        
        global_max_norm_sq = np.max(norms_sq)
        
        # Check if v is far enough from all selected
        if selected_vecs:
            sel_arr = np.array(selected_vecs)
            # Compute squared distances from v to all selected
            # (v - sel_arr)^2
            diff = v[None, :] - sel_arr
            dists_sq = np.sum(diff ** 2, axis=1)
            
            # Check if all distances are >= global_max_norm_sq
            if np.min(dists_sq) < global_max_norm_sq:
                continue
        
        # If we get here, v is valid
        selected_indices.append(idx)
        selected_vecs.append(v)
        
        # Early exit if we hit a good number? No, keep going until time.
        # But as set grows, checking gets slower.
        # If len(selected) > 500, maybe stop? 
        # The kissing number in 11D is likely in the hundreds? 
        # Actually, kissing number in d=11 is not known exactly, but bounds are ~500-1000?
        # Let's allow up to 1000.
        if len(selected_indices) >= 1000:
            break

    if not selected_indices:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result = cand_arr[selected_indices]
    return result

# EVOLVE-BLOCK-END
