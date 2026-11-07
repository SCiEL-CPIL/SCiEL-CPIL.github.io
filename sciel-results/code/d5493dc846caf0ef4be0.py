"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin from 90s limit

    # Precompute bit counts for all masks
    bit_counts = np.zeros(total_masks, dtype=np.int32)
    for mask in range(1, total_masks):
        bit_counts[mask] = bit_counts[mask & (mask - 1)] + 1

    # Precompute comparability matrix C
    # C[i, j] = 1 if masks[i] and masks[j] are comparable (one is strict subset of the other)
    # We use a boolean matrix. 4096x4096 is ~16MB.
    # To save memory/time, we can compute it on the fly or use a sparse representation?
    # 16MB is fine.
    
    # Efficient way to compute comparability:
    # i is comparable to j if (i | j) == i or (i | j) == j.
    # Let's create an array of masks.
    masks = np.arange(total_masks, dtype=np.uint16)
    
    # C will be uint8 for fast summing
    C = np.zeros((total_masks, total_masks), dtype=np.uint8)
    
    # We can vectorize the comparison.
    # For each i, we need to check all j.
    # This is O(N^2) which is 16M operations. Numpy can handle this quickly.
    
    # Expand masks to NxN
    # i_expanded = masks[:, np.newaxis]
    # j_expanded = masks[np.newaxis, :]
    # union = i_expanded | j_expanded
    # C = (union == i_expanded) | (union == j_expanded)
    # This creates two 4096x4096 arrays. Memory ~16MB each. Total ~48MB. Acceptable.
    
    i_expanded = masks[:, np.newaxis]
    j_expanded = masks[np.newaxis, :]
    union = i_expanded | j_expanded
    
    # Strict subset: A is strict subset of B if A|B == B and A != B.
    # Since we are looking for unordered pairs in the final score, and the sets are distinct,
    # "comparable" means (A|B)==A or (A|B)==B.
    # Note: A|A == A, so C[i,i] will be 1. We must handle this.
    # The problem asks for pairs where one is a STRICT subset of the other.
    # So for distinct i, j, if (i|j)==i or (i|j)==j, they are comparable.
    # If i==j, they are not strictly subset.
    
    comp_bool = (union == i_expanded) | (union == j_expanded)
    # Set diagonal to 0
    np.fill_diagonal(comp_bool, False)
    C = comp_bool.astype(np.uint8)

    def mask_to_list(mask):
        return [b for b in range(n) if (mask >> b) & 1]

    best_score = -1
    best_selection = []
    
    # Try multiple random initializations
    num_restarts = 100 # Can be higher now
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with a random chain
        perm = list(range(n))
        random.shuffle(perm)
        cur = 0
        selected_masks = [0]
        for p in perm:
            cur |= (1 << p)
            selected_masks.append(cur)
        
        # If we have more than m, truncate? No, chain length is n+1 = 13. m=40.
        # We need to fill more.
        
        selected_set = set(selected_masks)
        
        # Fill remaining slots greedily
        while len(selected_masks) < m:
            if time.time() > deadline:
                break
                
            # Calculate gain for all candidates not in selected_set
            # gain[cand] = sum(C[cand, sel] for sel in selected_masks)
            
            # We can use matrix multiplication or summing.
            # selected_indices = np.array(list(selected_set))
            # gains = C[:, selected_indices].sum(axis=1)
            # This is efficient.
            
            selected_indices = np.array(list(selected_set), dtype=np.int32)
            gains = C[:, selected_indices].sum(axis=1)
            
            # Mask out already selected
            gains[selected_indices] = -1
            
            best_candidate = np.argmax(gains)
            
            if gains[best_candidate] <= 0:
                # No more comparable candidates? Or we ran out of good ones.
                # If all gains are <= 0, we might as well pick a random one to fill the slot?
                # But we want to maximize score. If no candidate adds to score, adding it doesn't hurt score (delta 0).
                # So we can just pick any unselected.
                available = [i for i in range(total_masks) if i not in selected_set]
                if not available:
                    break
                best_candidate = random.choice(available)
            
            selected_masks.append(best_candidate)
            selected_set.add(best_candidate)
        
        # Calculate initial score
        # score = sum(C[i, j] for i in S, j in S, i < j)
        # Since C is symmetric and 0 on diagonal, score = 0.5 * sum(C[S, S])
        sel_arr = np.array(selected_masks, dtype=np.int32)
        sub_C = C[np.ix_(sel_arr, sel_arr)]
        current_score = int(np.sum(sub_C) / 2)
        
        if current_score > best_score:
            best_score = current_score
            best_selection = list(selected_masks)
        
        # --- Phase 2: Local Search (1-swap) ---
        # Maintain comp_count: number of selected items comparable to each candidate
        # comp_count[k] = sum(C[k, s] for s in S)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Update comp_count for current S
            sel_arr = np.array(selected_masks, dtype=np.int32)
            comp_count = C[:, sel_arr].sum(axis=1)
            
            for idx in range(len(selected_masks)):
                if time.time() >= deadline:
                    break
                
                u = selected_masks[idx]
                
                # Loss of removing u: comp_count[u]
                # (Since C[u,u]=0, comp_count[u] is exactly sum_{v in S, v!=u} C[u,v])
                
                # Gain of adding v (v not in S):
                # Sum_{w in S \ {u}} C[v, w]
                # If C[v, u] == 1, then Sum = comp_count[v] - 1
                # If C[v, u] == 0, then Sum = comp_count[v]
                
                # Delta(v) = Gain(v) - Loss(u)
                # Delta(v) = (comp_count[v] - C[v, u]) - comp_count[u]
                
                # We want to find v not in S that maximizes Delta(v)
                # Delta(v) = comp_count[v] - C[v, u] - comp_count[u]
                # Since comp_count[u] is constant for this u, we maximize comp_count[v] - C[v, u]
                
                # Vectorized calculation for all v
                # score_v = comp_count - C[:, u]
                score_v = comp_count - C[:, u]
                
                # Mask out selected items
                score_v[sel_arr] = -1
                
                best_v = np.argmax(score_v)
                
                # Check if improvement exists
                # Delta_best = score_v[best_v] - comp_count[u]
                delta_best = score_v[best_v] - comp_count[u]
                
                if delta_best > 0:
                    # Perform swap
                    selected_set.remove(u)
                    selected_set.add(best_v)
                    selected_masks[idx] = best_v
                    current_score += int(delta_best)
                    improved = True
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_selection = list(selected_masks)
                    
                    # Restart the sweep
                    break

    # Convert best_selection to lists of integers
    result = [mask_to_list(mask) for mask in best_selection]
    
    # Ensure types are correct: list of lists of int
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])
    
    return final_result
# EVOLVE-BLOCK-END
