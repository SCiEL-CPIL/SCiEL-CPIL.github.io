"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = P['n']
    m = P['m']
    num_masks = 1 << n
    
    # 1. Precompute comparability matrix
    masks = np.arange(num_masks, dtype=np.int32)
    
    # Using broadcasting
    mask_i = masks[:, None]
    mask_j = masks[None, :]
    subset_ij = (mask_i | mask_j) == mask_j
    subset_ji = (mask_i | mask_j) == mask_i
    not_equal = mask_i != mask_j
    comp = ((subset_ij | subset_ji) & not_equal).astype(np.int8)
    
    def local_search(selected_indices, comp, in_set, score_in_set, current_score, deadline, best_score, best_indices):
        """Perform single-swap and 2-swap local search."""
        current_set = set(selected_indices)
        
        # Single-swap local search
        while time.time() < deadline - 3:
            masked_scores = np.where(in_set, -1, score_in_set)
            max_val = int(np.max(masked_scores))
            max_idx = int(np.argmax(masked_scores))
            
            min_A_val = min(score_in_set[A] for A in selected_indices)
            
            if max_val - min_A_val <= 0:
                break
            
            best_delta = max_val - min_A_val
            best_out = min(selected_indices, key=lambda x: score_in_set[x])
            best_in = max_idx
            
            score_in_set += comp[:, best_in] - comp[:, best_out]
            
            selected_indices[selected_indices.index(best_out)] = best_in
            current_set.remove(best_out)
            current_set.add(best_in)
            in_set[best_out] = False
            in_set[best_in] = True
            
            current_score += best_delta
            if current_score > best_score[0]:
                best_score[0] = current_score
                best_indices[0] = list(selected_indices)
        
        # 2-swap local search
        while time.time() < deadline - 3:
            best_delta2 = 0
            best_out1 = -1
            best_out2 = -1
            best_in1 = -1
            best_in2 = -1
            
            # For each pair (A, B) in set, compute their combined contribution
            # When removing A and B: lose comp[A,B] + sum_{k in set\{A,B}} (comp[A,k] + comp[B,k])
            # = (score_in_set[A] - comp[A,B]) + (score_in_set[B] - comp[A,B]) + comp[A,B]
            # Wait, let me recalculate.
            # score_in_set[A] = sum_{k in set} comp[A,k] = comp[A,B] + sum_{k in set\{A,B}} comp[A,k]
            # So sum_{k in set\{A,B}} comp[A,k] = score_in_set[A] - comp[A,B]
            # Similarly for B.
            # Total lost = comp[A,B] + (score_in_set[A] - comp[A,B]) + (score_in_set[B] - comp[A,B])
            # = score_in_set[A] + score_in_set[B] - comp[A,B]
            
            # When adding C and D: gain = comp[C,D] + sum_{k in set\{A,B}} (comp[C,k] + comp[D,k])
            # But score_in_set[C] = sum_{k in set} comp[C,k] = comp[C,A] + comp[C,B] + sum_{k in set\{A,B}} comp[C,k]
            # So sum_{k in set\{A,B}} comp[C,k] = score_in_set[C] - comp[C,A] - comp[C,B]
            # Similarly for D.
            # Total gained = comp[C,D] + (score_in_set[C] - comp[C,A] - comp[C,B]) + (score_in_set[D] - comp[D,A] - comp[D,B])
            # = score_in_set[C] + score_in_set[D] + comp[C,D] - comp[C,A] - comp[C,B] - comp[D,A] - comp[D,B]
            
            # Delta = gained - lost
            # = score_in_set[C] + score_in_set[D] + comp[C,D] - comp[C,A] - comp[C,B] - comp[D,A] - comp[D,B] - score_in_set[A] - score_in_set[B] + comp[A,B]
            
            # For efficiency, iterate over candidate C, D not in set, and for each, find best A, B.
            # This is O(N^2 * m^2) which is too slow.
            
            # Alternative approach: for each C not in set, precompute adjusted_gain_C[A,B] = 
            #   score_in_set[C] - comp[C,A] - comp[C,B]
            # Then for each D, delta = adjusted_gain_C[A,B] + score_in_set[D] + comp[C,D] - comp[D,A] - comp[D,B] - score_in_set[A] - score_in_set[B] + comp[A,B]
            
            # Still complex. Let's use a simpler heuristic: try all pairs (C,D) not in set, and for each, find the best (A,B) to remove.
            # But that's O(N^2 * m^2).
            
            # Let's try a different approach: randomly sample candidate 2-swaps.
            num_samples = 500
            for _ in range(num_samples):
                if time.time() > deadline - 3:
                    break
                    
                # Sample two elements to remove
                out_pair = list(np.random.choice(len(selected_indices), size=2, replace=False))
                A = selected_indices[out_pair[0]]
                B = selected_indices[out_pair[1]]
                
                # Compute lost
                lost = score_in_set[A] + score_in_set[B] - comp[A, B]
                
                # Sample two elements to add
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) < 2:
                    continue
                in_pair = list(np.random.choice(len(not_in_set), size=2, replace=False))
                C = not_in_set[in_pair[0]]
                D = not_in_set[in_pair[1]]
                
                # Compute gained
                gained = score_in_set[C] + score_in_set[D] + comp[C, D] - comp[C, A] - comp[C, B] - comp[D, A] - comp[D, B]
                
                delta = gained - lost
                if delta > best_delta2:
                    best_delta2 = delta
                    best_out1 = A
                    best_out2 = B
                    best_in1 = C
                    best_in2 = D
            
            if best_delta2 <= 0:
                break
            
            # Apply 2-swap
            score_in_set += comp[:, best_in1] + comp[:, best_in2] - comp[:, best_out1] - comp[:, best_out2]
            
            selected_indices[selected_indices.index(best_out1)] = best_in1
            selected_indices[selected_indices.index(best_out2)] = best_in2
            current_set.remove(best_out1)
            current_set.remove(best_out2)
            current_set.add(best_in1)
            current_set.add(best_in2)
            in_set[best_out1] = False
            in_set[best_out2] = False
            in_set[best_in1] = True
            in_set[best_in2] = True
            
            current_score += best_delta2
            if current_score > best_score[0]:
                best_score[0] = current_score
                best_indices[0] = list(selected_indices)
        
        return selected_indices, in_set, score_in_set, current_score
    
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
        
    # 3. Local Search
    current_set = set(selected_indices)
    in_set = np.zeros(num_masks, dtype=bool)
    for idx in selected_indices:
        in_set[idx] = True
        
    score_in_set = comp[:, list(current_set)].sum(axis=1).astype(np.int32)
    
    best_score = [0]
    for idx in selected_indices:
        best_score[0] += score_in_set[idx]
    best_score[0] //= 2
    
    current_score = best_score[0]
    best_indices = [list(selected_indices)]
    
    # Run local search
    selected_indices, in_set, score_in_set, current_score = local_search(
        selected_indices, comp, in_set, score_in_set, current_score, deadline, best_score, best_indices
    )
    
    # Random restarts
    np.random.seed(42)
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization with some structure
        # Mix of random and greedy
        if restart % 2 == 0:
            # Pure random
            selected_indices = list(np.random.choice(num_masks, size=m, replace=False))
        else:
            # Greedy with random perturbation
            selected = np.zeros(num_masks, dtype=bool)
            selected_indices = []
            scores = comp.sum(axis=1).astype(np.int32)
            
            for _ in range(m):
                if time.time() > deadline - 5:
                    break
                if len(selected_indices) == 0:
                    best_idx = int(np.argmax(scores))
                else:
                    gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
                    gains[selected] = -1
                    # Add some randomness
                    gains += np.random.normal(0, 1, num_masks).astype(np.int32)
                    best_idx = int(np.argmax(gains))
                    
                selected_indices.append(best_idx)
                selected[best_idx] = True
            
            if len(selected_indices) < m:
                continue
        
        current_set = set(selected_indices)
        in_set = np.zeros(num_masks, dtype=bool)
        for idx in selected_indices:
            in_set[idx] = True
        
        score_in_set = comp[:, list(current_set)].sum(axis=1).astype(np.int32)
        
        restart_score = 0
        for idx in selected_indices:
            restart_score += score_in_set[idx]
        restart_score //= 2
        
        # Run local search
        selected_indices, in_set, score_in_set, restart_score = local_search(
            selected_indices, comp, in_set, score_in_set, restart_score, deadline, best_score, best_indices
        )
    
    # Final polish: try single-swap again on best solution
    if time.time() < deadline - 2:
        best_idx_list = best_indices[0]
        current_set = set(best_idx_list)
        in_set = np.zeros(num_masks, dtype=bool)
        for idx in best_idx_list:
            in_set[idx] = True
        
        score_in_set = comp[:, list(current_set)].sum(axis=1).astype(np.int32)
        
        polish_score = best_score[0]
        
        # Try single-swap polish
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            masked_scores = np.where(in_set, -1, score_in_set)
            max_val = int(np.max(masked_scores))
            max_idx = int(np.argmax(masked_scores))
            
            min_A_val = min(score_in_set[A] for A in best_idx_list)
            
            if max_val - min_A_val <= 0:
                break
            
            best_delta = max_val - min_A_val
            best_out = min(best_idx_list, key=lambda x: score_in_set[x])
            best_in = max_idx
            
            score_in_set += comp[:, best_in] - comp[:, best_out]
            
            best_idx_list[best_idx_list.index(best_out)] = best_in
            current_set.remove(best_out)
            current_set.add(best_in)
            in_set[best_out] = False
            in_set[best_in] = True
            
            polish_score += best_delta
            if polish_score > best_score[0]:
                best_score[0] = polish_score
                best_indices[0] = list(best_idx_list)
    
    # Convert to list of lists of Python ints
    result = []
    for mask in best_indices[0]:
        mask_int = int(mask)
        subset = [b for b in range(n) if (mask_int >> b) & 1]
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END
