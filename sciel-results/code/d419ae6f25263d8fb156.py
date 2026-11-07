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
    # C[i, j] is 1 if i and j are comparable (strict subset/superset), 0 otherwise
    C = np.zeros((total_subsets, total_subsets), dtype=np.int8)
    
    for i in range(total_subsets):
        subset_mask = (all_masks & i) == all_masks # all_masks subset of i
        superset_mask = (all_masks & i) == i # i subset of all_masks
        comparable = subset_mask | superset_mask
        comparable[i] = 0 # Exclude self
        C[i] = comparable.astype(np.int8)
    
    best_result = None
    best_score = -1
    
    # Try multiple random restarts
    num_restarts = 200
    for trial in range(num_restarts):
        random.seed(trial + 2000)
        
        # Greedy Initialization
        selected_set = set()
        selected_list = []
        
        # Start with a random set
        start = random.randint(0, total_subsets - 1)
        selected_set.add(start)
        selected_list.append(start)
        
        # Maintain gain array: gain[x] = sum of C[x, y] for y in selected_list
        gain = np.zeros(total_subsets, dtype=np.int32)
        # gain[start] is 0 since C[start, start] is 0
        
        for _ in range(m - 1):
            # Find best x not in selected_set
            # We want to maximize gain[x]
            # Mask out selected
            scores = gain.copy()
            for sel in selected_set:
                scores[sel] = -1
            
            best_idx = int(np.argmax(scores))
            selected_set.add(best_idx)
            selected_list.append(best_idx)
            
            # Update gain array
            # gain += C[:, best_idx]
            gain += C[:, best_idx]
        
        # Calculate current score
        # Sum of degrees = 2 * edges
        current_score = int(np.sum(gain) // 2)
        
        # Local Search Phase
        # We try to swap elements to improve score.
        # Maintain gain array accurately.
        
        for ls_iter in range(50):
            improved = False
            indices = list(range(m))
            random.shuffle(indices)
            
            for k in indices:
                mask_to_remove = selected_list[k]
                
                # Loss is gain[mask_to_remove]
                loss = int(gain[mask_to_remove])
                
                # Find best x not in selected_set to add
                # New gain for x if we remove mask_to_remove and add x:
                # The gain of x in the new set (S \ {u} + {x}) is:
                # (gain[x] - C[x, u])
                # We want to maximize this for x not in S.
                
                potential_gains = gain - C[:, mask_to_remove]
                
                # Exclude already selected
                for sel in selected_set:
                    potential_gains[sel] = -1
                    
                best_x = int(np.argmax(potential_gains))
                best_gain_val = int(potential_gains[best_x])
                
                net_gain = best_gain_val - loss
                
                if net_gain > 0:
                    # Perform swap
                    # Remove mask_to_remove, add best_x
                    selected_set.remove(mask_to_remove)
                    selected_set.add(best_x)
                    selected_list[k] = best_x
                    
                    # Update gain array
                    # gain changes by -C[:, mask_to_remove] + C[:, best_x]
                    gain -= C[:, mask_to_remove]
                    gain += C[:, best_x]
                    
                    current_score += net_gain
                    improved = True
                    # Break to restart the local search pass with updated gain
                    break
            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_result = selected_list.copy()
            
    # Convert selected masks to lists of integers
    result = []
    for mask in best_result:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result

# EVOLVE-BLOCK-END
