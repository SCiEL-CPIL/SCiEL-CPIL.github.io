import itertools
import random
import math
import time
import numpy as np
from scipy.sparse import csr_matrix

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute all k-subsets and their covered t-subset indices
    all_k_subsets = []
    block_t_indices = []
    
    for block in itertools.combinations(range(v), k):
        all_k_subsets.append(block)
        t_indices = []
        for ts in itertools.combinations(block, t):
            t_indices.append(t_subset_to_index[ts])
        block_t_indices.append(t_indices)
        
    num_k_subsets = len(all_k_subsets)
    
    # Build sparse matrix for fast greedy
    rows = []
    cols = []
    for i, indices in enumerate(block_t_indices):
        rows.extend([i] * len(indices))
        cols.extend(indices)
        
    M = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_k_subsets, num_t_subsets))
    
    # 1. Greedy Initialization
    uncovered_mask = np.ones(num_t_subsets, dtype=bool)
    current_blocks = []
    
    while np.any(uncovered_mask):
        gains = M.dot(uncovered_mask)
        best_block_idx = np.argmax(gains)
        max_gain = gains[best_block_idx]
        
        if max_gain == 0:
            break
            
        current_blocks.append(best_block_idx)
        indices = M[best_block_idx].indices
        uncovered_mask[indices] = False
        
    current_blocks = list(current_blocks)
    
    coverage_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in current_blocks:
        coverage_counts[M[b_idx].indices] += 1
        
    best_blocks = current_blocks[:]
    best_size = len(current_blocks)
    
    start_time = time.time()
    time_limit = 105.0
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Pass 1: Remove redundant blocks
        new_blocks = []
        for b_idx in current_blocks:
            indices = M[b_idx].indices
            if np.min(coverage_counts[indices]) > 1:
                coverage_counts[indices] -= 1
                improved = True
            else:
                new_blocks.append(b_idx)
        
        current_blocks = new_blocks
        
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_blocks = current_blocks[:]
            improved = True
            
        # Pass 2: Simple Swap Heuristic (if time allows and stuck)
        if not improved and (time.time() - start_time) < time_limit - 5:
            # Try a few random swaps
            swaps_tried = 0
            max_swaps = 100
            while swaps_tried < max_swaps and (time.time() - start_time) < time_limit - 5:
                swaps_tried += 1
                if len(current_blocks) < 2:
                    break
                
                # Pick a random block to potentially swap out
                pos = random.randint(0, len(current_blocks) - 1)
                b_old = current_blocks[pos]
                old_indices = M[b_old].indices
                
                # Find t-subsets that become uncovered if b_old is removed
                # These are indices where coverage_counts == 1
                critical_mask = (coverage_counts == 1)
                # The t-subsets that b_old covers AND are critical
                # We need a new block that covers ALL these critical t-subsets covered by b_old
                
                # Get the critical t-subsets covered by b_old
                # critical_indices_in_old = [i for i in old_indices if coverage_counts[i] == 1]
                # If empty, b_old is redundant (should have been caught in Pass 1, but maybe not if counts changed?)
                # Pass 1 runs before this, so if we are here, b_old has at least one critical t-subset.
                
                critical_ts = [i for i in old_indices if coverage_counts[i] == 1]
                
                if not critical_ts:
                    continue
                
                # Find a block b_new that covers all critical_ts
                # We can check this by seeing if the intersection of b_new's coverage and critical_ts is all of critical_ts.
                # Or simpler: For each candidate b_new, check if it covers all critical_ts.
                # To speed up, we can just check a sample or use the sparse matrix.
                # Let's just pick a random b_new and check.
                
                b_new = random.randint(0, num_k_subsets - 1)
                if b_new == b_old:
                    continue
                
                new_indices = M[b_new].indices
                # Check if b_new covers all critical_ts
                # Set intersection check
                # Convert critical_ts to a set? Or use numpy isin?
                # isin is fast for small arrays.
                if len(set(critical_ts).intersection(set(new_indices))) == len(critical_ts):
                    # b_new covers all critical parts.
                    # Apply swap
                    current_blocks[pos] = b_new
                    
                    # Update counts
                    coverage_counts[old_indices] -= 1
                    coverage_counts[new_indices] += 1
                    
                    # Check if this enables removals? 
                    # We don't know yet. We'll let Pass 1 handle it in the next iteration.
                    # But we should check if the solution is still valid (it should be, by construction).
                    # And if size is same, we haven't improved size yet.
                    # So we mark improved to trigger another Pass 1?
                    # Yes, because the new structure might have redundant blocks.
                    improved = True
                    break # Restart the loop with Pass 1

    final_blocks = [list(all_k_subsets[i]) for i in best_blocks]
    
    return final_blocks
