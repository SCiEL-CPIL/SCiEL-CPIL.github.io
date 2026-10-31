import itertools
import time

def construct(v, k, t):
        start_time = time.time()
        deadline = start_time + 115.0
        
        # 1. Precomputation
        all_tsubs = list(itertools.combinations(range(v), t))
        n_tsubs = len(all_tsubs)
        tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
        full_mask = (1 << n_tsubs) - 1
        
        all_ksubs = list(itertools.combinations(range(v), k))
        n_ksubs = len(all_ksubs)
        
        ksub_masks = [0] * n_ksubs
        for j, ks in enumerate(all_ksubs):
            mask = 0
            for tsub in itertools.combinations(ks, t):
                idx = tsub_to_idx[tsub]
                mask |= (1 << idx)
            ksub_masks[j] = mask
        
        # 2. Greedy Initialization
        selected_indices = []
        uncovered = full_mask
        
        while uncovered and time.time() < deadline - 5.0:
            best_gain = 0
            best_idx = -1
            for j in range(n_ksubs):
                if j in selected_indices: # Use set for O(1) lookup? 
                    continue
                gain = (uncovered & ksub_masks[j]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j
            
            if best_idx == -1:
                break
                
            selected_indices.append(best_idx)
            uncovered &= ~ksub_masks[best_idx]
            
            # Optimization: If uncovered becomes 0, break
            if uncovered == 0:
                break
        
        if not selected_indices:
            return []
            
        selected_set = set(selected_indices)
        
        # 3. Local Search: Remove and Swap
        # We want to minimize len(selected_indices)
        
        # Helper: Calculate union mask of a list of indices
        def get_union_mask(indices):
            union = 0
            for idx in indices:
                union |= ksub_masks[idx]
            return union
        
        # Helper: Check if a specific index is redundant given the set of indices
        # Redundant if Union(indices \ {idx}) == full_mask
        # We can compute Union(indices) and then check if removing idx breaks it?
        # No, Union(indices \ {idx}) is not simply Union(indices) \ mask(idx).
        # We must recompute or use counts.
        
        # Efficient Redundancy Check:
        # Precompute union of all selected.
        # For each i, compute union of all others.
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            n_sel = len(selected_indices)
            
            # Calculate union for all selected
            total_union = 0
            for idx in selected_indices:
                total_union |= ksub_masks[idx]
                
            # If not full cover, we can't remove anything. (Should not happen if greedy finished)
            if total_union != full_mask:
                 # Try to repair? Greedy should have covered it.
                 # If not, we are in trouble. Return what we have.
                 break
            
            # 3a. Try to remove redundant blocks
            new_selected = []
            for i, idx in enumerate(selected_indices):
                # Calculate union of all blocks EXCEPT idx
                # Optimization: If we do this for every i, it's O(N^2).
                # N is ~140. 140*140 = 20k ops. Fast.
                
                # Alternative: Maintain prefix/suffix unions?
                # Simpler: Just loop.
                union_others = 0
                for j, other_idx in enumerate(selected_indices):
                    if i != j:
                        union_others |= ksub_masks[other_idx]
                
                if union_others == full_mask:
                    # idx is redundant
                    improved = True
                    # Do not add idx to new_selected
                else:
                    new_selected.append(idx)
            
            selected_indices = new_selected
            selected_set = set(selected_indices)
            
            # 3b. If no removal happened, try swaps
            if not improved and len(selected_indices) > 0:
                swapped = False
                # For each block in selected, try to find a replacement
                # that allows a subsequent removal?
                # Or just find a swap that results in a valid cover, then retry removal.
                
                # To limit cost, we might only do a few swaps.
                # Let's try to find ANY swap.
                
                for i, idx_out in enumerate(selected_indices):
                    if time.time() > deadline - 2.0:
                        break
                    
                    # Union of all others
                    union_others = 0
                    for j, idx in enumerate(selected_indices):
                        if i != j:
                            union_others |= ksub_masks[idx]
                    
                    missing = full_mask & ~union_others
                    
                    if missing == 0:
                        # This block is redundant, should have been caught above
                        continue
                        
                    # Find b_in not in selected such that mask(b_in) covers missing
                    # mask(b_in) & missing == missing
                    
                    # Iterate all ksubs
                    # Optimization: Skip if gain is 0?
                    # We just need to cover 'missing'.
                    
                    found_swap = False
                    for j in range(n_ksubs):
                        if j in selected_set:
                            continue
                        
                        # Check if j covers missing
                        if (ksub_masks[j] & missing) == missing:
                            # Valid swap
                            selected_indices[i] = j
                            selected_set.remove(idx_out)
                            selected_set.add(j)
                            swapped = True
                            found_swap = True
                            break
                            
                    if found_swap:
                        # Restart the outer loop to check for new redundancies
                        improved = True
                        break # Break inner loop, restart outer
                        
                if not swapped:
                    # No swaps found, stop
                    break
            
        # Convert back to list of lists
        return [list(all_ksubs[idx]) for idx in selected_indices]
