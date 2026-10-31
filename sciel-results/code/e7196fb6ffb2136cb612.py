import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precomputation
    # Map t-subsets to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    full_mask = (1 << n_tsubs) - 1
    
    # Generate all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # Precompute masks for each k-subset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask
        
    # Convert to numpy array for faster bitwise operations if possible? 
    # Python ints are arbitrary precision, so bitwise ops on large ints are fast.
    # numpy uint64 would limit us to 64 t-subsets, which is far too small (C(19,4)=3876).
    # So we stick with Python ints.
    
    def get_union_mask(indices):
        union = 0
        for idx in indices:
            union |= ksub_masks[idx]
        return union

    def is_valid_cover(selected_indices):
        return get_union_mask(selected_indices) == full_mask

    # 2. Greedy Initialization
    selected_indices = []
    uncovered = full_mask
    
    while uncovered and time.time() < deadline - 10.0:
        best_gain = 0
        best_idx = -1
        for j in range(n_ksubs):
            if j in selected_indices:
                continue
            gain = (uncovered & ksub_masks[j]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = j
        
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        uncovered &= ~ksub_masks[best_idx]
        
        if uncovered == 0:
            break
            
    if not selected_indices:
        return []
        
    # Ensure we have a valid cover. If greedy didn't finish, try to fix or return best effort.
    # But greedy should always finish for this problem size.
    
    best_solution = selected_indices[:]
    best_size = len(best_solution)
    
    # 3. Local Search: 1-for-1 Swaps and Redundancy Removal
    # We iterate until no improvement is found or time runs out.
    
    # To speed up "is this block redundant?", we can maintain counts of how many blocks cover each t-subset.
    # But with 3876 t-subsets and ~150 blocks, recomputing union is O(N * M) where M is bit ops.
    # 150 * 3876 is small. The bottleneck is the search for swaps.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        n_sel = len(selected_indices)
        
        # 3a. Remove redundant blocks
        # Check each block: if removing it still covers everything, remove it.
        # We do this repeatedly in a pass.
        new_selected = []
        for i, idx in enumerate(selected_indices):
            # Compute union of all other blocks
            # Optimization: Precompute total union, then if we remove idx, 
            # we can't just subtract. We must recompute or use counts.
            # Let's use counts for efficiency in the removal phase?
            # Actually, for N~150, O(N^2) recomputation is 22500 ops. Very fast.
            
            union_others = 0
            for j, other_idx in enumerate(selected_indices):
                if i != j:
                    union_others |= ksub_masks[other_idx]
            
            if union_others == full_mask:
                # Redundant
                improved = True
                # Do not add to new_selected
            else:
                new_selected.append(idx)
                
        selected_indices = new_selected
        
        if len(selected_indices) < best_size:
            best_size = len(selected_indices)
            best_solution = selected_indices[:]
        
        # 3b. If no removals, try swaps
        if not improved and len(selected_indices) > 0:
            selected_set = set(selected_indices)
            swapped = False
            
            # Try to find a swap for each block
            # For each block i, compute what is uniquely covered by it (missing if removed)
            # Then find a block j not in selected that covers 'missing'.
            
            for i, idx_out in enumerate(selected_indices):
                if time.time() > deadline - 5.0:
                    break
                    
                # Compute union of all others
                union_others = 0
                for j, idx in enumerate(selected_indices):
                    if i != j:
                        union_others |= ksub_masks[idx]
                
                missing = full_mask & ~union_others
                
                if missing == 0:
                    # Should have been removed in 3a, but maybe we missed it?
                    # Or it's just redundant.
                    continue
                    
                # Find a replacement
                # We want a block j such that (ksub_masks[j] & missing) == missing
                # i.e., ksub_masks[j] covers all the 'missing' parts.
                
                # Optimization: Pre-filter?
                # Just iterate.
                found_swap = False
                for j in range(n_ksubs):
                    if j in selected_set:
                        continue
                    
                    # Check if j covers missing
                    if (ksub_masks[j] & missing) == missing:
                        # Valid swap
                        # Perform swap
                        selected_indices[i] = j
                        selected_set.remove(idx_out)
                        selected_set.add(j)
                        swapped = True
                        found_swap = True
                        
                        # Update best if size decreased? No, size is same.
                        # But we might enable further reductions.
                        break
                        
                if found_swap:
                    improved = True
                    # Restart the outer loop to check for new redundancies
                    break # Break inner loop, restart outer
                    
            if not swapped:
                # No swaps found, stop
                break

    # Final check
    if is_valid_cover(selected_indices) and len(selected_indices) < best_size:
        best_solution = selected_indices[:]
        
    return [list(all_ksubs[idx]) for idx in best_solution]
