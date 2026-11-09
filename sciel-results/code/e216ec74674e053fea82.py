import numpy as np
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_vectors = []
    best_score = 0
    
    # Generate candidate pool
    # 1. Norm squared 2: (±1, ±1, 0, ...)
    candidates = []
    
    # Add all vectors of type (±1, ±1, 0, ...)
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    # 2. Norm squared 4: (±2, 0, ...)
    for i in range(D):
        for s in [-2, 2]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # 3. Norm squared 4: (±1, ±1, ±1, ±1, 0, ...) - Optional, might be too many
    # Let's stick to the above first. If time permits, add more.
    
    # Convert to array for vectorized ops
    # candidates is a list of arrays. Stack them.
    if not candidates:
         return np.array([], dtype=np.int64).reshape(0, D)
         
    cand_array = np.stack(candidates)
    num_cands = len(candidates)
    
    # Greedy Search with Vectorization
    # We will try multiple random orders
    
    # Pre-compute norms squared of candidates
    cand_norms_sq = np.sum(cand_array * cand_array, axis=1)
    
    # We want to maximize k.
    # The constraint is min pairwise dist >= max norm.
    # Let R2_max be the max norm squared among selected.
    # For any new v, dist_sq to all u in S must be >= R2_max.
    
    # Strategy:
    # If we only select norm 2 vectors, R2_max = 2.
    # If we select a norm 4 vector, R2_max = 4.
    # It is generally easier to pack norm 2 vectors if we keep R2_max low.
    # However, norm 4 vectors are "sparser" in a sense? No, (2,0..) is very sparse.
    # Let's try a mixed approach:
    # 1. Start with all norm 4 axis vectors? No, they conflict with many norm 2.
    # 2. Just do a greedy on the mixed pool, but prioritize adding vectors that don't increase R2_max.
    
    # Actually, a simpler robust strategy:
    # Run greedy on ONLY norm 2 vectors. R2_max = 2.
    # Then try to add norm 4 vectors that fit.
    # Or just run greedy on the whole pool, and track R2_max.
    
    def greedy_search(order, max_iters=1000):
        selected = []
        selected_array = []
        R2_max = 0
        
        for idx in order:
            if time.time() > deadline - 2:
                break
            if len(selected) >= max_iters:
                break
                
            v = cand_array[idx]
            v_norm_sq = cand_norms_sq[idx]
            
            # New R2_max if we add v
            new_R2_max = max(R2_max, v_norm_sq)
            
            if not selected:
                selected.append(idx)
                selected_array.append(v)
                R2_max = new_R2_max
                continue
            
            # Check distance to all selected
            # dist_sq = ||v||^2 + ||u||^2 - 2 v.u
            # We need dist_sq >= new_R2_max for all u
            
            # Vectorized dot products
            # selected_array is a list of 1D arrays. Stack for speed if large.
            # For small selected, loop is fine. For large, stack.
            
            if len(selected_array) < 100:
                valid = True
                for u in selected_array:
                    dist_sq = v_norm_sq + np.dot(u, u) - 2 * np.dot(v, u)
                    if dist_sq < new_R2_max:
                        valid = False
                        break
                if valid:
                    selected.append(idx)
                    selected_array.append(v)
                    R2_max = new_R2_max
            else:
                # Stack selected for matrix mult
                sel_stack = np.stack(selected_array)
                dots = sel_stack @ v
                sel_norms_sq = np.sum(sel_stack * sel_stack, axis=1)
                dist_sqs = v_norm_sq + sel_norms_sq - 2 * dots
                if np.min(dist_sqs) >= new_R2_max:
                    selected.append(idx)
                    selected_array.append(v)
                    R2_max = new_R2_max
                    
        return selected_array, R2_max
    
    # Try multiple random orders
    # We prioritize orders that put norm 2 vectors first?
    # Or just random.
    
    indices = np.arange(num_cands)
    
    # Sort indices by norm squared (ascending) to prefer smaller norms initially?
    # Smaller norms allow for denser packing if R2_max stays low.
    # But if we mix, it's complex.
    # Let's try:
    # 1. Order sorted by norm (norm 2 first)
    # 2. Random orders
    
    trial_orders = []
    
    # Order 1: Norm 2 first
    sorted_indices = indices[np.argsort(cand_norms_sq)]
    trial_orders.append(sorted_indices)
    
    # Order 2-10: Random
    for _ in range(9):
        random_idx = indices.copy()
        np.random.shuffle(random_idx)
        trial_orders.append(random_idx)
        
    for order in trial_orders:
        if time.time() > deadline - 5:
            break
            
        res_vectors, _ = greedy_search(order)
        
        if len(res_vectors) > best_score:
            best_score = len(res_vectors)
            best_vectors = res_vectors.copy()
            
    if not best_vectors:
        # Fallback
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return np.array(best_vectors, dtype=np.int64)
