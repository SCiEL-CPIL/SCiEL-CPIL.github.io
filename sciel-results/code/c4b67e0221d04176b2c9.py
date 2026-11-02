import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, store the indices of t-subsets it contains
    block_t_indices = []
    # Also store for each t-subset, the list of blocks containing it (for local search)
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        t_indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            t_indices.append(idx)
            t_subset_to_blocks[idx].append(b_idx)
        block_t_indices.append(t_indices)
    
    # Convert block_t_indices to a 2D numpy array for vectorized operations
    # All blocks have exactly C(k, t) t-subsets, so this is rectangular
    block_matrix = np.array(block_t_indices, dtype=np.int32)

    def time_check():
        return time.time() > deadline

    # 2. Greedy Initialization (Vectorized)
    # We want to cover all t-subsets.
    # Objective: Maximize number of newly covered t-subsets.
    counts = np.zeros(num_t_subsets, dtype=np.int32)
    blocks = []
    
    remaining_uncovered = num_t_subsets
    
    while remaining_uncovered > 0:
        if time_check():
            break
            
        # Vectorized scoring:
        # counts[block_matrix] has shape (num_blocks, C(k,t))
        # We count how many entries in each row are 0 (uncovered)
        covered_mask = counts[block_matrix] == 0
        new_covs = np.sum(covered_mask, axis=1)
        
        best_block_idx = np.argmax(new_covs)
        max_new_cov = new_covs[best_block_idx]
        
        if max_new_cov == 0:
            # No block covers any new t-subset, but we have remaining?
            # This implies a bug or invalid parameters, but break to avoid infinite loop
            break
            
        # Add the best block
        best_block = int(best_block_idx)
        blocks.append(best_block)
        counts[block_matrix[best_block]] += 1
        remaining_uncovered -= max_new_cov
        
    # 2b. Force Cover: Ensure validity if greedy was interrupted
    if remaining_uncovered > 0:
        if not time_check():
            uncovered_indices = np.where(counts == 0)[0]
            for idx in uncovered_indices:
                if time_check():
                    break
                candidates = t_subset_to_blocks[idx]
                if candidates:
                    # Pick a block that might cover other uncovered subsets?
                    # For speed, just pick the first one or random
                    b_idx = candidates[0]
                    if b_idx not in blocks:
                        blocks.append(b_idx)
                        counts[block_matrix[b_idx]] += 1
            # Recalculate remaining to be sure
            remaining_uncovered = np.sum(counts == 0)

    # 3. Local Search (Pruning and Swapping)
    # Goal: Minimize len(blocks)
    
    def remove_redundant():
        """Remove all redundant blocks in one pass."""
        nonlocal blocks, counts
        new_blocks = []
        removed_count = 0
        for b_idx in blocks:
            bti = block_matrix[b_idx]
            if np.all(counts[bti] > 1):
                # Redundant
                counts[bti] -= 1
                removed_count += 1
            else:
                new_blocks.append(b_idx)
        if removed_count > 0:
            blocks = new_blocks
        return removed_count > 0

    def single_swap():
        """Try to remove a block and re-cover its unique t-subsets with one other block."""
        nonlocal blocks, counts
        if not blocks:
            return False
        
        random.shuffle(blocks)
        
        for b_out_idx in range(len(blocks)):
            if time_check():
                return False
                
            b_out = blocks[b_out_idx]
            bti_out = block_matrix[b_out]
            # Find t-subsets unique to b_out
            lost_mask = counts[bti_out] == 1
            lost_indices = bti_out[lost_mask]
            
            if len(lost_indices) == 0:
                # Redundant
                counts[bti_out] -= 1
                blocks.pop(b_out_idx)
                return True
            
            # Find a block b_in that covers all lost_indices
            candidate_blocks = None
            for idx in lost_indices:
                if candidate_blocks is None:
                    candidate_blocks = set(t_subset_to_blocks[idx])
                else:
                    candidate_blocks.intersection_update(t_subset_to_blocks[idx])
                
                if not candidate_blocks:
                    break
            
            if not candidate_blocks:
                continue
            
            # Exclude b_out itself
            candidate_blocks.discard(b_out)
            
            if not candidate_blocks:
                continue
            
            # Pick a random candidate
            b_in = random.choice(list(candidate_blocks))
            
            # Perform swap
            counts[bti_out] -= 1
            blocks.pop(b_out_idx)
            
            if b_in not in blocks:
                bti_in = block_matrix[b_in]
                counts[bti_in] += 1
                blocks.append(b_in)
            
            return True
        
        return False

    def pairwise_swap():
        """Try to replace 2 blocks with 1 block, or 1 block with 2 blocks (net zero but may enable further removals)."""
        nonlocal blocks, counts
        if len(blocks) < 2:
            return False
        
        # Try to find two blocks whose union of unique t-subsets can be covered by one new block
        n = len(blocks)
        random.shuffle(blocks)
        
        # Limit pairs to check for time efficiency
        max_pairs = min(500, n * (n - 1) // 2)
        pairs_checked = 0
        
        for i in range(n):
            if time_check() or pairs_checked >= max_pairs:
                return False
                
            b_out1 = blocks[i]
            bti1 = block_matrix[b_out1]
            
            for j in range(i + 1, n):
                if time_check() or pairs_checked >= max_pairs:
                    return False
                pairs_checked += 1
                
                b_out2 = blocks[j]
                bti2 = block_matrix[b_out2]
                
                # Find t-subsets unique to b_out1 or b_out2 (count == 1)
                lost1 = bti1[counts[bti1] == 1]
                lost2 = bti2[counts[bti2] == 1]
                
                # Combined lost set
                if len(lost1) == 0 and len(lost2) == 0:
                    # Both redundant
                    counts[bti1] -= 1
                    counts[bti2] -= 1
                    blocks.pop(j)
                    blocks.pop(i)
                    return True
                
                # Find a block that covers all lost1 and lost2
                all_lost = np.unique(np.concatenate([lost1, lost2])) if len(lost1) > 0 and len(lost2) > 0 else (lost1 if len(lost1) > 0 else lost2)
                
                if len(all_lost) == 0:
                    continue
                
                candidate_blocks = None
                for idx in all_lost:
                    if candidate_blocks is None:
                        candidate_blocks = set(t_subset_to_blocks[idx])
                    else:
                        candidate_blocks.intersection_update(t_subset_to_blocks[idx])
                    
                    if not candidate_blocks:
                        break
                
                if not candidate_blocks:
                    continue
                
                # Exclude b_out1 and b_out2
                candidate_blocks.discard(b_out1)
                candidate_blocks.discard(b_out2)
                
                if not candidate_blocks:
                    continue
                
                # Pick a random candidate
                b_in = random.choice(list(candidate_blocks))
                
                # Perform swap: remove b_out1 and b_out2, add b_in
                counts[bti1] -= 1
                counts[bti2] -= 1
                blocks.pop(j)
                blocks.pop(i)
                
                if b_in not in blocks:
                    bti_in = block_matrix[b_in]
                    counts[bti_in] += 1
                    blocks.append(b_in)
                
                return True
        
        return False

    def local_search():
        nonlocal blocks, counts
        improved = True
        while improved and not time_check():
            improved = False
            
            # Pass 1: Remove redundant blocks
            if remove_redundant():
                improved = True
            
            if time_check():
                break
            
            # Pass 2: Single swap
            if single_swap():
                improved = True
            
            if time_check():
                break
            
            # Pass 3: Pairwise swap
            if pairwise_swap():
                improved = True
        
        return blocks

    local_search()
    
    # 4. Final polish: run local search again with more aggressive settings
    if not time_check():
        # Run additional passes
        for _ in range(5):
            if time_check():
                break
            if remove_redundant() or single_swap() or pairwise_swap():
                continue
            else:
                break
    
    # Convert indices back to lists
    result = [list(all_blocks[b_idx]) for b_idx in blocks]
    
    return result
