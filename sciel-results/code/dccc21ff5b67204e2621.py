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
    masks = np.arange(num_masks, dtype=np.int32)
    
    mask_i = masks[:, None]
    mask_j = masks[None, :]
    subset_ij = (mask_i | mask_j) == mask_j
    subset_ji = (mask_i | mask_j) == mask_i
    not_equal = mask_i != mask_j
    comp = ((subset_ij | subset_ji) & not_equal).astype(np.int8)
    
    best_overall_score = 0
    best_overall_indices = None
    
    rng = np.random.RandomState(42)
    
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # 2. Greedy Initialization with variation
        selected = np.zeros(num_masks, dtype=bool)
        selected_indices = []
        
        strategy = restart % 3
        
        if strategy == 0:
            for _ in range(m):
                if time.time() > deadline - 15:
                    break
                if len(selected_indices) == 0:
                    scores = comp.sum(axis=1).astype(np.int32)
                    best_idx = int(np.argmax(scores))
                else:
                    gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
                    gains[selected] = -1
                    best_idx = int(np.argmax(gains))
                selected_indices.append(best_idx)
                selected[best_idx] = True
                
        elif strategy == 1:
            scores = comp.sum(axis=1).astype(np.int32)
            for _ in range(m):
                if time.time() > deadline - 15:
                    break
                if len(selected_indices) == 0:
                    best_idx = int(np.argmax(scores))
                else:
                    gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
                    gains[selected] = -1
                    noise = rng.random(num_masks) * 0.5
                    gains_noisy = gains + noise
                    gains_noisy[selected] = -1
                    best_idx = int(np.argmax(gains_noisy))
                selected_indices.append(best_idx)
                selected[best_idx] = True
                
        else:
            candidates = np.arange(num_masks)
            rng.shuffle(candidates)
            for idx in candidates[:m]:
                selected_indices.append(int(idx))
                selected[int(idx)] = True
        
        # 3. Local Search
        in_set = np.zeros(num_masks, dtype=bool)
        for idx in selected_indices:
            in_set[idx] = True
            
        score_in_set = comp[:, list(selected_indices)].sum(axis=1).astype(np.int32)
        
        current_score = int(score_in_set[list(selected_indices)].sum() // 2)
        
        # Single-swap local search (best improvement)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            masked_scores = np.where(in_set, -1, score_in_set)
            max_val = int(np.max(masked_scores))
            max_idx = int(np.argmax(masked_scores))
            
            if max_val < 0:
                break
            
            # Find min score in set
            min_score = np.inf
            min_idx = -1
            for A in selected_indices:
                if score_in_set[A] < min_score:
                    min_score = score_in_set[A]
                    min_idx = A
            
            best_delta = max_val - min_score
            if best_delta <= 0:
                break
            
            # Apply swap
            score_in_set += comp[:, max_idx] - comp[:, min_idx]
            selected_indices[selected_indices.index(min_idx)] = max_idx
            in_set[min_idx] = False
            in_set[max_idx] = True
            current_score += best_delta
            improved = True
        
        # 2-swap local search (best improvement)
        # For fixed A in set, C not in set:
        # delta(A,B,C,D) = [sC - cCA - cCB] + [sD - cDA - cDB] + cCD - [sA - cAB] - [sB - cBA]
        # where sX = score_in_set[X], cXY = comp[X,Y]
        # = (sC - cCA) + (sD - cDA) + cCD - cCB - cDB - sA + cAB - sB + cBA
        # = (sC - cCA - sA) + (sD - cDA) + cCD - cCB - cDB - sB + cAB + cBA
        # Note cAB = cBA (symmetric)
        # = (sC - cCA - sA + 2*cAB) + (sD - cDA - sB) + cCD - cCB - cDB
        
        # For fixed A, C: constant_part = sC - cCA - sA
        # Then we need to find B in S\{A}, D in N\{C} to maximize:
        # constant_part + 2*cAB + (sD - cDA - sB) + cCD - cCB - cDB
        
        # Precompute for each C not in set: sC, and for each B in set: sB
        # For each (A, C) pair, iterate over B in S\{A} and for each B, find best D
        
        # This is O(m * N * m * N) = O(m^2 * N^2) which is too slow for full search.
        # Instead, sample: for each A in set, try top-K C's not in set by (sC - cCA)
        
        improved2 = True
        while improved2 and time.time() < deadline - 8:
            improved2 = False
            best_delta_2 = 0
            best_A = -1
            best_B = -1
            best_C = -1
            best_D = -1
            
            # For each A in set
            for A in selected_indices:
                if time.time() > deadline - 8:
                    break
                
                # Get candidates C not in set, sorted by potential
                # sC - cCA is the main term
                c_A = comp[A, :]  # row A of comp
                potential_C = np.where(in_set, -1, score_in_set - c_A)
                
                # Get top K candidates for C
                K = 50
                top_C_indices = np.argsort(potential_C)[-K:][::-1]
                top_C_indices = [int(c) for c in top_C_indices if potential_C[c] >= 0]
                
                if not top_C_indices:
                    continue
                
                for C in top_C_indices:
                    if time.time() > deadline - 8:
                        break
                    
                    # For this (A, C), find best (B, D)
                    # constant = score_in_set[C] - comp[C,A] - score_in_set[A]
                    constant = score_in_set[C] - comp[C, A] - score_in_set[A]
                    
                    # For each B in S\{A}:
                    for B in selected_indices:
                        if B == A:
                            continue
                        
                        c_AB = comp[A, B]
                        c_CB = comp[C, B]
                        
                        # For each D not in set, D != C:
                        # term = constant + 2*c_AB + (score_in_set[D] - comp[D,A] - score_in_set[B]) + comp[C,D] - c_CB - comp[D,B]
                        # = (constant + 2*c_AB - c_CB) + (score_in_set[D] - comp[D,A] - score_in_set[B] + comp[C,D] - comp[D,B])
                        
                        base_B = constant + 2 * c_AB - c_CB
                        
                        # Find best D not in set, D != C
                        # score_in_set[D] - comp[D,A] - score_in_set[B] + comp[C,D] - comp[D,B]
                        # = (score_in_set[D] - comp[D,A] - comp[D,B]) + comp[C,D] - score_in_set[B]
                        
                        # Precompute for this B:
                        # For all D not in set: val[D] = score_in_set[D] - comp[D,A] - comp[D,B] + comp[C,D]
                        # Then best D maximizes val[D] - score_in_set[B]
                        
                        # Compute val for all D not in set
                        in_set_no_C = in_set.copy()
                        in_set_no_C[C] = False  # D can be C? No, D != C
                        # Actually D should not be in set and D != C
                        # Let's just compute for all D not in set, then exclude C
                        
                        # val[D] = score_in_set[D] - comp[D,A] - comp[D,B] + comp[C,D]
                        # = score_in_set[D] - comp[D,A] - comp[D,B] + comp[C,D]
                        
                        # This is O(N) per B, and we have O(m) B's, O(K) C's, O(m) A's
                        # Total: O(m * K * m * N) = O(m^2 * K * N)
                        # m=40, K=50, N=4096: 40*50*40*4096 = 327,680,000 which might be tight
                        
                        # Let's reduce K to 20
                        if len(top_C_indices) > 20:
                            top_C_indices = top_C_indices[:20]
                        
                        # Compute val for all D
                        # val[D] = score_in_set[D] - comp[D,A] - comp[D,B] + comp[C,D]
                        val_D = score_in_set - comp[:, A] - comp[:, B] + comp[C, :]
                        val_D[in_set] = -1000  # D must not be in set
                        val_D[C] = -1000  # D != C
                        
                        best_D = int(np.argmax(val_D))
                        best_val_D = int(val_D[best_D])
                        
                        if best_val_D < -500:
                            continue
                        
                        total_delta = base_B + best_val_D - score_in_set[B]
                        
                        if total_delta > best_delta_2:
                            best_delta_2 = total_delta
                            best_A = A
                            best_B = B
                            best_C = C
                            best_D = best_D
            
            if best_delta_2 <= 0:
                break
            
            # Apply 2-swap
            score_in_set += comp[:, best_C] + comp[:, best_D] - comp[:, best_A] - comp[:, best_B]
            
            # Update selected_indices
            idx_A = selected_indices.index(best_A)
            idx_B = selected_indices.index(best_B)
            selected_indices[idx_A] = best_C
            selected_indices[idx_B] = best_D
            
            in_set[best_A] = False
            in_set[best_B] = False
            in_set[best_C] = True
            in_set[best_D] = True
            
            current_score += best_delta_2
            improved2 = True
        
        # Update best
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_indices = list(selected_indices)
    
    # Convert to list of lists of Python ints
    if best_overall_indices is None:
        result = []
        for mask in selected_indices:
            mask_int = int(mask)
            subset = [b for b in range(n) if (mask_int >> b) & 1]
            result.append(subset)
        return result
    
    result = []
    for mask in best_overall_indices:
        mask_int = int(mask)
        subset = [b for b in range(n) if (mask_int >> b) & 1]
        result.append(subset)
    
    return result
# EVOLVE-BLOCK-END
