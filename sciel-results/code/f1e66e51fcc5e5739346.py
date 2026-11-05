import itertools
import time
import random
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety
    
    # 1. Precomputation with NumPy
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map t-subset to index
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence matrix: cover_matrix[b, i] = 1 if block b covers t_subset i
    # Shape: (num_blocks, num_t_subsets)
    # Using uint8 for memory efficiency and speed
    cover_matrix = np.zeros((num_blocks, num_t_subsets), dtype=np.uint8)
    
    for b_idx, block in enumerate(all_blocks):
        # Generate all t-subsets within this block
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            cover_matrix[b_idx, idx] = 1
            
    # Helper: Greedy construction using NumPy
    def greedy_construct():
        # uncovered: bool array of shape (num_t_subsets,)
        # Initially all True
        uncovered = np.ones(num_t_subsets, dtype=bool)
        selected_blocks = []
        
        # Precompute block sums to speed up initial check? Not strictly necessary if we use argmax on product.
        # gain = sum(cover_matrix[b] * uncovered)
        # We want argmax_b gain.
        
        while np.any(uncovered):
            # Calculate gain for all blocks
            # cover_matrix is (num_blocks, num_t_subsets)
            # uncovered is (num_t_subsets,)
            # gain = cover_matrix @ uncovered (if uncovered is 0/1)
            # Since uncovered is bool, we can cast to float or use logical operations.
            # Using dot product with float is fast.
            gains = cover_matrix.dot(uncovered.astype(np.float32))
            
            best_block_idx = np.argmax(gains)
            best_gain = gains[best_block_idx]
            
            if best_gain == 0:
                # Should not happen if problem is solvable, but safety check
                break
            
            selected_blocks.append(int(best_block_idx))
            
            # Update uncovered
            # uncovered = uncovered & ~cover_matrix[best_block_idx]
            # cover_matrix[best_block_idx] is uint8, need to cast to bool
            uncovered &= ~cover_matrix[best_block_idx].astype(bool)
            
            # Check time
            if time.time() > deadline - 5:
                break
                
        return selected_blocks

    # Helper: Reduction using NumPy
    def reduce_blocks(block_indices):
        if not block_indices:
            return []
            
        # Count how many times each t-subset is covered by the selected blocks
        # subset_counts[i] = number of blocks in block_indices covering t_subset i
        subset_counts = np.sum(cover_matrix[block_indices], axis=0)
        
        # A block b can be removed if for all t-subsets i covered by b, subset_counts[i] > 1
        # i.e., min(subset_counts[i] for i where cover_matrix[b, i] == 1) > 1
        
        changed = True
        current_indices = list(block_indices)
        
        while changed:
            changed = False
            new_indices = []
            
            # We need to check each block in current_indices
            # To make this fast, we can vectorize the check if possible, but indices change.
            # However, num_selected is small (~300). num_blocks is large.
            # We only check blocks in current_indices.
            
            for b_idx in current_indices:
                # Find t-subsets covered by this block
                covered_by_b = cover_matrix[b_idx] == 1
                # Get counts for these t-subsets
                counts_for_b = subset_counts[covered_by_b]
                
                # If any count is 1, we cannot remove this block
                if np.min(counts_for_b) > 1:
                    # Safe to remove
                    # Update subset_counts
                    subset_counts -= cover_matrix[b_idx]
                    changed = True
                else:
                    new_indices.append(b_idx)
                    
            current_indices = new_indices
            
            # Check time
            if time.time() > deadline - 5:
                break
                
        return current_indices

    # Main Logic
    best_solution = None
    best_size = float('inf')
    
    # 1. Run standard greedy
    if time.time() < deadline - 20:
        sol = greedy_construct()
        sol = reduce_blocks(sol)
        best_solution = sol
        best_size = len(sol)
        
    # 2. Local Search / Swap
    # We try to swap blocks to enable further reduction.
    if best_solution is not None and time.time() < deadline - 10:
        current_solution = list(best_solution)
        # Precompute mask for current solution for fast swap checks?
        # We can recompute counts for the current solution.
        subset_counts = np.sum(cover_matrix[current_solution], axis=0)
        
        # Set of indices for O(1) lookup
        sol_set = set(current_solution)
        
        # We will perform random swaps
        # Pick a random block in solution, pick a random block outside, swap if valid.
        # Valid if new subset_counts still >= 1 for all.
        
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            if not current_solution:
                break
                
            # Pick random block in solution
            old_block = random.choice(current_solution)
            
            # Pick random block outside solution
            # To avoid expensive rejection sampling, we can pick from a precomputed list of non-solution blocks?
            # Or just pick random and check. Since solution is small (~300) vs total (~75k), rejection is rare.
            new_block = random.randint(0, num_blocks - 1)
            if new_block in sol_set:
                continue
            
            # Check validity of swap
            # new_counts = subset_counts - cover_matrix[old] + cover_matrix[new]
            # We need min(new_counts) >= 1
            
            # Optimization: Only check if the swap actually changes the coverage in a way that might break it.
            # If old_block and new_block have identical masks, swap is trivially valid (but useless for reduction).
            # If new_block covers all that old_block covers, it's valid.
            
            # Fast check:
            # Remove old, add new
            # We can simulate:
            # temp_counts = subset_counts - cover_matrix[old_block]
            # if np.any(temp_counts < 0): continue (shouldn't happen if old is in solution)
            # new_counts = temp_counts + cover_matrix[new_block]
            # if np.any(new_counts == 0): continue
            
            # To be faster, we can just check the intersection.
            # Actually, direct NumPy ops are fast enough for a few thousand iterations.
            
            temp_counts = subset_counts - cover_matrix[old_block]
            # If any count becomes 0, we must ensure new_block covers it.
            # Uncovered_by_old = temp_counts == 0
            # If new_block doesn't cover any of these, invalid.
            # Also, if new_block covers something else, that's fine.
            
            # Simpler:
            new_counts = temp_counts + cover_matrix[new_block]
            
            if np.min(new_counts) >= 1:
                # Valid swap
                # Apply
                current_solution[current_solution.index(old_block)] = new_block
                sol_set.remove(old_block)
                sol_set.add(new_block)
                subset_counts = new_counts
                
                # Try to reduce
                new_sol = reduce_blocks(current_solution)
                if len(new_sol) < len(current_solution):
                    current_solution = new_sol
                    sol_set = set(current_solution)
                    # Recalculate counts
                    subset_counts = np.sum(cover_matrix[current_solution], axis=0)
                    
                    if len(current_solution) < best_size:
                        best_size = len(current_solution)
                        best_solution = current_solution.copy()
                        
                # If no improvement, we keep the swap? 
                # Keeping the swap might lead to a better state later, but it increases risk of wandering.
                # Standard local search usually reverts if no improvement.
                # Let's revert to maintain a locally minimal state.
                # Revert swap
                current_solution[current_solution.index(new_block)] = old_block
                sol_set.remove(new_block)
                sol_set.add(old_block)
                subset_counts = np.sum(cover_matrix[current_solution], axis=0)
            else:
                pass # Invalid swap, do nothing

    return [list(all_blocks[b_idx]) for b_idx in best_solution]
