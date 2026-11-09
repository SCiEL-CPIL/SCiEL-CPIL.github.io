"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = P['n']
    m = P['m']
    num_masks = 1 << n
    
    # 1. Precompute comparability matrix
    masks = np.arange(num_masks, dtype=np.int32)
    mask_i = masks[:, None]
    mask_j = masks[None, :]
    subset_ij = (mask_i | mask_j) == mask_j
    subset_ji = (mask_i | mask_j) == mask_i
    not_equal = mask_i != mask_j
    comp = ((subset_ij | subset_ji) & not_equal).astype(np.int8)
    
    # 2. Greedy Initialization
    selected = np.zeros(num_masks, dtype=bool)
    selected_indices = []
    scores = comp.sum(axis=1).astype(np.int32)
    
    for _ in range(m):
        if time.time() > deadline - 10:
            break
        if len(selected_indices) == 0:
            best_idx = int(np.argmax(scores))
        else:
            gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
            gains[selected] = -1
            best_idx = int(np.argmax(gains))
        selected_indices.append(best_idx)
        selected[best_idx] = True
        
    # 3. Local Search with Simulated Annealing
    current_set = set(selected_indices)
    in_set = np.zeros(num_masks, dtype=bool)
    for idx in selected_indices:
        in_set[idx] = True
        
    score_in_set = comp[:, list(current_set)].sum(axis=1).astype(np.int32)
    
    # Calculate initial score
    current_score = 0
    for idx in selected_indices:
        current_score += score_in_set[idx]
    current_score //= 2
    
    best_score = current_score
    best_indices = list(selected_indices)
    
    # Simulated annealing parameters
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.995
    
    T = T_start
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 5 and T > T_end:
        # Find best swap (deterministic)
        # Precompute masked scores once
        masked_scores = np.where(in_set, -1, score_in_set)
        max_val = np.max(masked_scores)
        max_idx = int(np.argmax(masked_scores))
        
        best_delta = 0
        best_out = -1
        best_in = max_idx
        
        for A in selected_indices:
            delta = max_val - score_in_set[A]
            if delta > best_delta:
                best_delta = delta
                best_out = A
        
        if best_delta > 0:
            # Always accept improving moves
            # Apply swap
            score_in_set += comp[:, best_in] - comp[:, best_out]
            selected_indices[selected_indices.index(best_out)] = best_in
            current_set.remove(best_out)
            current_set.add(best_in)
            in_set[best_out] = False
            in_set[best_in] = True
            current_score += best_delta
            if current_score > best_score:
                best_score = current_score
                best_indices = list(selected_indices)
        else:
            # No improving move: try random swap with probability based on temperature
            # Pick a random element to remove and a random element to add
            A = int(rng.choice(selected_indices))
            # Find candidates not in set
            not_in_mask = ~in_set
            candidates = np.where(not_in_mask)[0]
            if len(candidates) > 0:
                B = int(rng.choice(candidates))
                delta = score_in_set[B] - score_in_set[A]
                # Accept if improving or with probability exp(-delta/T) for worsening
                if delta >= 0 or rng.random() < np.exp(-delta / T):
                    # Apply swap
                    score_in_set += comp[:, B] - comp[:, A]
                    selected_indices[selected_indices.index(A)] = B
                    current_set.remove(A)
                    current_set.add(B)
                    in_set[A] = False
                    in_set[B] = True
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_indices = list(selected_indices)
        
        T *= cooling_rate
    
    # Final polish: run best-improvement local search until convergence
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        masked_scores = np.where(in_set, -1, score_in_set)
        max_val = np.max(masked_scores)
        max_idx = int(np.argmax(masked_scores))
        
        best_delta = 0
        best_out = -1
        best_in = max_idx
        
        for A in selected_indices:
            delta = max_val - score_in_set[A]
            if delta > best_delta:
                best_delta = delta
                best_out = A
        
        if best_delta > 0:
            score_in_set += comp[:, best_in] - comp[:, best_out]
            selected_indices[selected_indices.index(best_out)] = best_in
            current_set.remove(best_out)
            current_set.add(best_in)
            in_set[best_out] = False
            in_set[best_in] = True
            current_score += best_delta
            if current_score > best_score:
                best_score = current_score
                best_indices = list(selected_indices)
            improved = True
    
    # Convert to list of lists of Python ints
    result = []
    for mask in best_indices:
        mask_int = int(mask)
        subset = [b for b in range(n) if (mask_int >> b) & 1]
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END
