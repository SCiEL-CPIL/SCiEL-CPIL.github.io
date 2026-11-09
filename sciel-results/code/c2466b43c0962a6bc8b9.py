"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = P['n']
    m = P['m']
    num_masks = 1 << n
    
    # 1. Precompute comparability matrix
    # comp[i, j] = 1 if i and j are comparable (one strict subset of other)
    masks = np.arange(num_masks, dtype=np.int32)
    # To save memory and time, we can compute on the fly or use a smaller representation?
    # 4096x4096 int8 is 16MB. This is fine.
    
    # Efficient construction of comp
    # i is subset of j iff (i | j) == j
    # i != j
    # We can do this in blocks to avoid huge temporary arrays if needed, but 16MB is small.
    
    # However, vectorized ops on 4096x4096 might be slow if done repeatedly.
    # We only do it once.
    
    # Using broadcasting
    mask_i = masks[:, None]
    mask_j = masks[None, :]
    subset_ij = (mask_i | mask_j) == mask_j
    subset_ji = (mask_i | mask_j) == mask_i
    not_equal = mask_i != mask_j
    comp = ((subset_ij | subset_ji) & not_equal).astype(np.int8)
    
    # 2. Greedy Initialization
    selected = np.zeros(num_masks, dtype=bool)
    selected_indices = []
    
    # Greedy: pick set with max degree, then next max degree to selected, etc.
    # To speed up, we can just pick random or simple greedy.
    # Simple greedy:
    scores = comp.sum(axis=1).astype(np.int32)
    
    for _ in range(m):
        if time.time() > deadline - 10:
            break
        # Pick argmax of scores, among not selected
        # To do this efficiently, we can just find max in scores where not selected
        # But scores changes? No, in greedy init, we don't update scores dynamically in the most optimal way (which would be O(N^2)), 
        # but simple greedy is fine for initialization.
        # Actually, let's do a slightly better greedy:
        # Calculate marginal gain for each unselected set.
        
        if len(selected_indices) == 0:
            best_idx = int(np.argmax(scores))
        else:
            # Marginal gain for x = sum(comp[x, k] for k in selected_indices)
            # This is O(N * m)
            # We can compute this vectorized
            # comp[:, selected_indices].sum(axis=1)
            gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
            gains[selected] = -1
            best_idx = int(np.argmax(gains))
            
        selected_indices.append(best_idx)
        selected[best_idx] = True
        
    # 3. Local Search (Best Improvement Swap)
    current_set = set(selected_indices)
    in_set = np.zeros(num_masks, dtype=bool)
    for idx in selected_indices:
        in_set[idx] = True
        
    # score_in_set[x] = sum(comp[x, k] for k in current_set)
    # Initialize
    score_in_set = comp[:, list(current_set)].sum(axis=1).astype(np.int32)
    
    best_score = 0
    # Calculate initial score
    # Sum of comp[i, j] for i < j in set
    # Or sum(score_in_set[i] for i in set) / 2
    for idx in selected_indices:
        best_score += score_in_set[idx]
    best_score //= 2
    
    current_score = best_score
    best_indices = list(selected_indices)
    
    while time.time() < deadline - 5:
        # Find best swap
        best_delta = 0
        best_out = -1
        best_in = -1
        
        # Iterate over all A in set
        for A in selected_indices:
            # Iterate over all B not in set
            # To speed up, we can find max score_in_set[B] for B not in set
            # max_B = max(score_in_set[~in_set])
            # Then delta = max_B - score_in_set[A]
            # If delta > best_delta, update
            
            # This is O(N) to find max.
            # Doing this for each A is O(m * N).
            # m=40, N=4096. 160k ops.
            
            # We can precompute max_B once per iteration?
            # No, score_in_set changes, but within the search for best swap, it's fixed.
            
            max_B_val = -1
            best_B_idx = -1
            
            # Find max score_in_set among not in set
            # We can use np.max on a mask, but creating a masked array is slow.
            # Instead, iterate? No, vectorized is better.
            # score_in_set_not_in = np.where(in_set, -1, score_in_set)
            # max_val = np.max(score_in_set_not_in)
            # max_idx = np.argmax(score_in_set_not_in)
            
            # This creates a temporary array of size N.
            # 40 times per iteration. 40 * 4096 = 160k elements processed.
            
            masked_scores = np.where(in_set, -1, score_in_set)
            max_val = np.max(masked_scores)
            max_idx = int(np.argmax(masked_scores))
            
            delta = max_val - score_in_set[A]
            
            if delta > best_delta:
                best_delta = delta
                best_out = A
                best_in = max_idx
                
        if best_delta <= 0:
            break # No improvement possible
            
        # Apply swap
        # Remove best_out, add best_in
        
        # Update score_in_set
        # score_in_set[x] += comp[x, best_in] - comp[x, best_out]
        score_in_set += comp[:, best_in] - comp[:, best_out]
        
        # Update set structures
        selected_indices[selected_indices.index(best_out)] = best_in
        current_set.remove(best_out)
        current_set.add(best_in)
        in_set[best_out] = False
        in_set[best_in] = True
        
        current_score += best_delta
        if current_score > best_score:
            best_score = current_score
            best_indices = list(selected_indices)
            
    # Convert to list of lists of Python ints
    result = []
    for mask in best_indices:
        mask_int = int(mask)
        subset = [b for b in range(n) if (mask_int >> b) & 1]
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END
