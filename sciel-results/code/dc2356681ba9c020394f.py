import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_targets = len(t_subsets)
    
    # 2. Generate all k-blocks and compute their coverage masks
    k_blocks = list(itertools.combinations(range(v), k))
    coverage_masks = []
    
    for block in k_blocks:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        coverage_masks.append(mask)
    
    # 3. Greedy selection using bitmasks
    uncovered_mask = (1 << n_targets) - 1
    selected_indices = []
    n_blocks = len(k_blocks)
    
    while uncovered_mask:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_gain = 0
        
        for i in range(n_blocks):
            gain = (coverage_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered_mask.bit_count():
                    break
        
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        uncovered_mask &= ~coverage_masks[best_idx]
    
    # 4. Deletion pass: remove redundant blocks
    def is_redundant(idx, sel_indices, cov_masks):
        """Check if block idx is redundant given the other selected blocks."""
        if idx not in sel_indices:
            return False
        # Compute union of all other selected blocks' masks
        union_mask = 0
        for j in sel_indices:
            if j != idx:
                union_mask |= cov_masks[j]
        # Block idx is redundant if all its covered targets are also covered by others
        return (cov_masks[idx] & ~union_mask) == 0
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(selected_indices) - 1, -1, -1):
            if time.time() > deadline:
                break
            if is_redundant(selected_indices[i], selected_indices, coverage_masks):
                selected_indices.pop(i)
                changed = True
    
    # 5. Swap-based local search: try replacing blocks to enable further deletions
    def try_swap(sel_indices, cov_masks, k_blocks, deadline):
        """Try to improve the solution by swapping one block with another."""
        best_swap = None
        best_size = len(sel_indices)
        
        # Try each selected block for replacement
        for i in range(len(sel_indices)):
            if time.time() > deadline:
                break
            orig_idx = sel_indices[i]
            
            # Try replacing with other blocks
            # Sample a subset of candidates for speed
            candidates = list(range(n_blocks))
            random.shuffle(candidates)
            candidates = candidates[:min(200, n_blocks)]
            
            for new_idx in candidates:
                if new_idx == orig_idx or new_idx in sel_indices:
                    continue
                
                # Tentatively swap
                new_sel = sel_indices[:]
                new_sel[i] = new_idx
                
                # Check if this enables any deletions
                # First, verify coverage is still complete
                union_mask = 0
                for j in new_sel:
                    union_mask |= cov_masks[j]
                
                if union_mask != (1 << n_targets) - 1:
                    continue
                
                # Try to delete redundant blocks from new_sel
                new_sel_copy = new_sel[:]
                improved = True
                while improved:
                    improved = False
                    for j in range(len(new_sel_copy) - 1, -1, -1):
                        if time.time() > deadline:
                            return None
                        if is_redundant(new_sel_copy[j], new_sel_copy, cov_masks):
                            new_sel_copy.pop(j)
                            improved = True
                
                if len(new_sel_copy) < best_size:
                    best_size = len(new_sel_copy)
                    best_swap = new_sel_copy
                    break
        
        return best_swap
    
    # Run several rounds of swap search
    for _ in range(5):
        if time.time() > deadline:
            break
        result = try_swap(selected_indices, coverage_masks, k_blocks, deadline)
        if result is not None:
            selected_indices = result
            # After a successful swap, run deletion again
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(selected_indices) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    if is_redundant(selected_indices[i], selected_indices, coverage_masks):
                        selected_indices.pop(i)
                        changed = True
        else:
            break
    
    return [list(k_blocks[i]) for i in selected_indices]
