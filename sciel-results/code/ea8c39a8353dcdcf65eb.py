import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Build dense coverage matrix M (num_blocks x num_t)
    M = np.zeros((num_blocks, num_t), dtype=np.uint8)
    
    for i, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            M[i, idx] = 1

    def greedy_set_cover(M, uncovered, selected_indices, deadline):
        """Greedy set cover with time check"""
        while uncovered.any():
            if time.time() > deadline:
                break
            
            # Calculate gain for each block
            gains = M @ uncovered
            
            if gains.max() == 0:
                break
            
            # Find block with max gain
            best_idx = np.argmax(gains)
            
            selected_indices.append(best_idx)
            
            # Update uncovered
            M_row = M[best_idx]
            uncovered[M_row == 1] = 0
        
        return selected_indices

    def redundancy_removal(M, selected_indices, deadline):
        """Remove redundant blocks from the selected set"""
        if not selected_indices:
            return []
        
        selected_indices = np.array(selected_indices, dtype=np.int64)
        
        # Compute coverage counts
        M_selected = M[selected_indices]
        coverage_counts = M_selected.sum(axis=0)
        
        keep_mask = np.ones(len(selected_indices), dtype=bool)
        
        for i in range(len(selected_indices)):
            if not keep_mask[i]:
                continue
            
            if time.time() > deadline:
                break
            
            block_row = M[selected_indices[i]]
            counts_of_covered = coverage_counts[block_row == 1]
            
            if len(counts_of_covered) > 0 and np.min(counts_of_covered) == 1:
                # Essential: some t-subset is only covered by this block
                continue
            else:
                # Redundant
                keep_mask[i] = False
                coverage_counts -= block_row
        
        return selected_indices[keep_mask]

    def local_search(M, selected_indices, deadline, num_restarts=10):
        """Local search to improve the covering"""
        if not selected_indices:
            return selected_indices
        
        best_solution = list(selected_indices)
        best_len = len(best_solution)
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            current = list(best_solution)
            
            # Try to remove blocks
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                
                improved = False
                M_selected = M[np.array(current, dtype=np.int64)]
                coverage_counts = M_selected.sum(axis=0)
                
                for i in range(len(current) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    
                    block_row = M[current[i]]
                    counts_of_covered = coverage_counts[block_row == 1]
                    
                    if len(counts_of_covered) == 0 or np.min(counts_of_covered) > 1:
                        # This block is redundant
                        current.pop(i)
                        coverage_counts -= block_row
                        improved = True
            
            # Try to add and remove blocks (swap improvement)
            if len(current) < best_len:
                best_solution = list(current)
                best_len = len(current)
            elif len(current) == best_len and restart < num_restarts // 2:
                # Try some swaps
                if time.time() > deadline - 5.0:
                    break
                
                M_selected = M[np.array(current, dtype=np.int64)]
                coverage_counts = M_selected.sum(axis=0)
                
                # Find blocks that cover uncovered or barely-covered t-subsets
                for i in range(len(current)):
                    if time.time() > deadline - 2.0:
                        break
                    
                    block_row_i = M[current[i]]
                    # Check if we can replace current[i] with a better block
                    counts_without_i = coverage_counts - block_row_i
                    if np.any(counts_without_i == 0):
                        # current[i] is essential, skip
                        continue
                    
                    # Try to find a replacement block
                    # A replacement should cover at least as many currently covered t-subsets
                    # and not leave any t-subset uncovered
                    candidate_blocks = np.where(M @ np.maximum(counts_without_i, 1) > 0)[0]
                    if len(candidate_blocks) == 0:
                        continue
                    
                    # Find candidate that maximizes coverage of barely-covered t-subsets
                    barely_covered = (counts_without_i == 1)
                    for candidate in candidate_blocks:
                        if candidate in current:
                            continue
                        
                        cand_row = M[candidate]
                        if np.any(cand_row[~barely_covered] == 1):
                            # Candidate covers some t-subset that would be uncovered
                            # unless we check more carefully
                            pass
                        
                        # Check if candidate is better
                        new_counts = counts_without_i + cand_row
                        if np.all(new_counts >= 1):
                            # Valid replacement
                            # Check if it improves anything (e.g., reduces max coverage)
                            current[i] = candidate
                            coverage_counts = new_counts
                            break
        
        return best_solution

    # Initial greedy cover
    uncovered = np.ones(num_t, dtype=np.uint8)
    selected_indices = greedy_set_cover(M, uncovered, [], deadline - 20.0)
    
    # Redundancy removal
    selected_indices = redundancy_removal(M, selected_indices, deadline - 15.0)
    
    # Local search
    selected_indices = local_search(M, selected_indices, deadline - 5.0, num_restarts=5)
    
    final_indices = np.array(selected_indices, dtype=np.int64) if selected_indices else np.array([], dtype=np.int64)
    final_blocks = [list(all_blocks[i]) for i in final_indices]
    
    return final_blocks
