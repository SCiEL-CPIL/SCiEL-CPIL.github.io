"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    n = 12
    m = 40
    total_subsets = 1 << n
    
    # Precompute subset masks as numpy array
    all_masks = np.arange(total_subsets, dtype=np.int32)
    
    # Precompute comparability matrix
    # comp_matrix[i, j] is 1 if i and j are comparable (strict subset/superset), 0 otherwise
    # Note: i and i are not comparable (strict), so diagonal is 0.
    comp_matrix = np.zeros((total_subsets, total_subsets), dtype=np.int8)
    
    # Efficiently compute comparability
    # A & B == A implies A subset B.
    # We can vectorize this.
    # For each i, find j such that (i & j) == i or (i & j) == j.
    # This is still O(N^2) if done naively.
    # But N=4096. 16M operations.
    # Let's try a loop with vectorization inside.
    for i in range(total_subsets):
        # j such that i is subset of j: (i & j) == i
        # j such that j is subset of i: (i & j) == j
        # We can compute masks for all j at once.
        # But i changes.
        # Actually, we can just do it in Python loops for the outer, vectorized for inner?
        # Or just a double loop if optimized?
        # 4096 * 4096 = 16M. Python double loop is ~1s. Acceptable.
        pass
    
    # Let's do it properly with NumPy broadcasting if possible, or just a fast loop.
    # Given the small size, a pure Python double loop might be borderline.
    # Let's use the property:
    # comparable(i, j) = 1 if (i & j) in {i, j} and i != j.
    
    # We can precompute this matrix.
    # To speed up, we can use the fact that comparability is transitive? No.
    
    # Let's just compute it once.
    # Using a loop over i, and vectorized check for j.
    for i in range(total_subsets):
        # Mask for j where i is subset of j: (i & j) == i
        # This is equivalent to: j must have all bits of i.
        # We can generate all supersets of i? No, we check all j.
        # (all_masks & i) == i  -> i is subset of all_masks
        # (all_masks & i) == all_masks -> all_masks is subset of i
        
        subset_mask = (all_masks & i) == all_masks # all_masks subset of i
        superset_mask = (all_masks & i) == i # i subset of all_masks
        comparable = subset_mask | superset_mask
        comparable[i] = 0 # Exclude self
        comp_matrix[i] = comparable.astype(np.int8)
    
    best_result = None
    best_score = -1
    
    # Try multiple random restarts
    num_restarts = 100
    for trial in range(num_restarts):
        random.seed(trial + 1000)
        selected_set = set()
        selected_list = []
        
        # Greedy Phase
        # Start with random
        start = random.randint(0, total_subsets - 1)
        selected_set.add(start)
        selected_list.append(start)
        
        for _ in range(m - 1):
            # Sum of comparable counts from selected_list
            # comp_matrix[selected_list] has shape (k, N)
            # Sum along axis 0
            scores = comp_matrix[selected_list].sum(axis=0)
            
            # Exclude selected
            for sel in selected_set:
                scores[sel] = -1
            
            best_idx = int(np.argmax(scores))
            selected_set.add(best_idx)
            selected_list.append(best_idx)
        
        # Calculate current score
        # Sum of upper triangle of comp_matrix[selected_list]
        sub_matrix = comp_matrix[selected_list][:, selected_list]
        # We want sum of i<j.
        # Since matrix is symmetric, sum(sub_matrix) / 2
        current_score = int(sub_matrix.sum() // 2)
        
        # Local Search Phase
        # We try to swap elements to improve score.
        # To make it fast, we use the precomputed matrix.
        
        for ls_iter in range(20):
            improved = False
            # Shuffle indices
            indices = list(range(m))
            random.shuffle(indices)
            
            for k in indices:
                mask_to_remove = selected_list[k]
                
                # Loss: number of comparable pairs involving mask_to_remove
                # This is the sum of the row (or col) for mask_to_remove in the sub_matrix
                # sub_matrix[k, :] sum
                loss = int(sub_matrix[k].sum())
                
                # Gain from adding a new mask x
                # We need count of comparable pairs between x and (selected_list \ {k})
                # Let counts_x be the number of elements in selected_list comparable to x.
                # counts_x = comp_matrix[x, selected_list].sum()
                # But we must exclude the pair with mask_to_remove.
                # So gain_x = counts_x - comp_matrix[x, mask_to_remove]
                
                # We want to find x not in selected_set that maximizes gain_x.
                # We can compute counts for all x in one go?
                # comp_matrix[:, selected_list] has shape (N, m)
                # Sum along axis 1 -> counts for all x
                counts_all = comp_matrix[:, selected_list].sum(axis=1)
                
                # Subtract the connection to mask_to_remove
                # comp_matrix[:, mask_to_remove] is a vector of length N
                gain_all = counts_all - comp_matrix[:, mask_to_remove]
                
                # Exclude already selected
                for sel in selected_set:
                    gain_all[sel] = -1
                
                best_x = int(np.argmax(gain_all))
                best_gain = int(gain_all[best_x])
                
                net_gain = best_gain - loss
                
                if net_gain > 0:
                    # Perform swap
                    selected_list[k] = best_x
                    selected_set.remove(mask_to_remove)
                    selected_set.add(best_x)
                    current_score += net_gain
                    improved = True
                    # Update sub_matrix? 
                    # It's easier to just recalculate sub_matrix after the swap for the next iterations
                    # Or we can update it incrementally.
                    # For simplicity and safety, let's just break and recalculate if needed?
                    # Actually, sub_matrix is used for loss calculation.
                    # If we swap, sub_matrix changes.
                    # We can just rebuild sub_matrix at the start of each ls_iter?
                    # No, we are inside the k loop.
                    # Let's just mark improved and break the k loop, then recalculate sub_matrix at the top of ls_iter.
                    break
            
            if not improved:
                break
            
            # Recalculate sub_matrix for next iteration
            sub_matrix = comp_matrix[selected_list][:, selected_list]
        
        if current_score > best_score:
            best_score = current_score
            best_result = selected_list.copy()
            
            # Early stop if we reach a very high score?
            # Max is 780.
            if best_score > 600:
                pass # Keep going to see if we can get higher
        
    # Convert selected masks to lists of integers
    result = []
    for mask in best_result:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result

# EVOLVE-BLOCK-END
