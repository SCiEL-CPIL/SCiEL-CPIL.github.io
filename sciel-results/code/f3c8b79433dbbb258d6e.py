import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    hard_deadline = 145.0  # Safety margin before 150s kill
    
    def time_check():
        return time.time() - start_time >= hard_deadline

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the indices of the t-subsets it covers
    # Using lists for fast iteration
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # Precompute degrees
    t_subset_degrees = [len(lst) for lst in t_subset_blocks]

    # 2. Phase 1: Weighted Greedy Independent Set (Packing)
    # Select blocks that do not share any t-subset.
    # Weight: Prefer blocks covering rare t-subsets.
    
    pool = set(range(num_k_subsets))
    core_blocks = []
    uncovered_mask = [1] * num_t_subsets
    
    # Initial weights: sum of 1/degree for t-subsets in block
    block_weights = [0.0] * num_k_subsets
    for b_idx, indices in enumerate(block_t_indices):
        w = 0.0
        for t_idx in indices:
            deg = t_subset_degrees[t_idx]
            if deg > 0:
                w += 1.0 / deg
        block_weights[b_idx] = w

    while pool:
        if time_check():
            break
        best_block = -1
        max_score = -1.0
        
        # Iterate pool to find best block
        # Optimization: If pool is very large, we might sample, but 50k is okay.
        # We can also pre-filter if possible, but dynamic scores make it hard.
        for b_idx in pool:
            cov_uncovered = 0
            scarcity_sum = 0.0
            for t_idx in block_t_indices[b_idx]:
                if uncovered_mask[t_idx]:
                    cov_uncovered += 1
                    scarcity_sum += 1.0 / t_subset_degrees[t_idx]
            
            if cov_uncovered == 0:
                continue
                
            score = cov_uncovered * 100.0 + scarcity_sum
            if score > max_score:
                max_score = score
                best_block = b_idx
                
        if best_block == -1:
            break
            
        core_blocks.append(best_block)
        
        for t_idx in block_t_indices[best_block]:
            uncovered_mask[t_idx] = 0
            
        neighbors_to_remove = set()
        for t_idx in block_t_indices[best_block]:
            neighbors_to_remove.update(t_subset_blocks[t_idx])
        
        pool.difference_update(neighbors_to_remove)

    # 3. Phase 2: Solve Remaining Set Cover
    active_blocks = set(core_blocks)
    cover_count = [0] * num_t_subsets
    
    for b_idx in active_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    still_uncovered = [i for i in range(num_t_subsets) if cover_count[i] == 0]
    
    if still_uncovered:
        residual_uncovered = set(still_uncovered)
        
        while residual_uncovered:
            if time_check():
                break
            best_block = -1
            max_gain = 0
            
            # To speed up, we can iterate over blocks that cover at least one residual item.
            # But we don't have a direct mapping from residual set to blocks efficiently without scanning.
            # Scanning 50k blocks * 35 t-subsets is ~1.75M ops. Acceptable per iteration if few iterations.
            # However, residual_uncovered shrinks.
            
            # Optimization: Only consider blocks NOT in active_blocks?
            # Yes, adding an active block does nothing.
            for b_idx in range(num_k_subsets):
                if b_idx in active_blocks:
                    continue
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if t_idx in residual_uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
                    
            if best_block == -1 or max_gain == 0:
                break
                
            active_blocks.add(best_block)
            for t_idx in block_t_indices[best_block]:
                if t_idx in residual_uncovered:
                    residual_uncovered.remove(t_idx)
                cover_count[t_idx] += 1

    # 4. Minimize
    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
            if time_check():
                return active_blocks_set, cover_counts
            changed = False
            items = list(active_blocks_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_blocks_set:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks_set.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_counts[t_idx] -= 1
                    changed = True
        return active_blocks_set, cover_counts

    active_blocks, cover_count = minimize(active_blocks, cover_count)
    best_size = len(active_blocks)
    best_blocks_copy = set(active_blocks)
    best_count_copy = list(cover_count)

    # 5. Local Search (Enhanced with Bitmask-like optimizations where possible)
    # We use standard set operations but optimized for Python.
    
    # Pre-compute block_t_indices as sets for faster intersection checks?
    # block_t_indices is list of lists. Let's convert to sets for swap checks if needed,
    # but checking `t_idx in block_t_indices[new_idx]` is O(35).
    
    while time.time() - start_time < 140.0:  # Leave 10s for final cleanup
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)
        
        for B in current_blocks:
            if B not in active_blocks:
                continue
            
            # Try Remove
            can_remove = True
            for t_idx in block_t_indices[B]:
                if cover_count[t_idx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                active_blocks.remove(B)
                for t_idx in block_t_indices[B]:
                    cover_count[t_idx] -= 1
                improved = True
                # Minimize after removal to catch cascading redundancies
                active_blocks, cover_count = minimize(active_blocks, cover_count)
                if len(active_blocks) < best_size:
                    best_size = len(active_blocks)
                    best_blocks_copy = set(active_blocks)
                    best_count_copy = list(cover_count)
                continue # B is gone
            
            # Try Swap
            # Find unique subsets of B
            unique_subsets = []
            for t_idx in block_t_indices[B]:
                if cover_count[t_idx] == 1:
                    unique_subsets.append(t_idx)
            
            if unique_subsets:
                # Intersect candidate lists
                candidates = None
                for t_idx in unique_subsets:
                    blocks_list = t_subset_blocks[t_idx]
                    if candidates is None:
                        candidates = set(blocks_list)
                    else:
                        candidates.intersection_update(blocks_list)
                    if not candidates:
                        break
                
                if candidates:
                    valid_candidates = [c for c in candidates if c != B and c not in active_blocks]
                    if valid_candidates:
                        # Pick one randomly
                        B_new = random.choice(valid_candidates)
                        
                        # Check if swap is valid (it is by construction of candidates covering uniques)
                        # But we must ensure B_new doesn't introduce conflicts? No, set cover allows overlaps.
                        # We just swap B for B_new.
                        
                        # Perform Swap
                        active_blocks.remove(B)
                        active_blocks.add(B_new)
                        
                        for t_idx in block_t_indices[B]:
                            cover_count[t_idx] -= 1
                        for t_idx in block_t_indices[B_new]:
                            cover_count[t_idx] += 1
                        
                        # Minimize after swap
                        active_blocks, cover_count = minimize(active_blocks, cover_count)
                        
                        if len(active_blocks) < best_size:
                            best_size = len(active_blocks)
                            best_blocks_copy = set(active_blocks)
                            best_count_copy = list(cover_count)
                            improved = True
                        else:
                            # If no improvement, revert? 
                            # The instruction says "keep the best result".
                            # If we don't revert, we might wander. 
                            # But "minimize" might have removed other blocks, so size might be same or smaller.
                            # If size is same, we might keep it for diversity? 
                            # Let's only keep if strictly smaller or if we want to explore.
                            # Given the goal is minimization, let's only keep if smaller.
                            # If not smaller, revert to best? Or current?
                            # The current program logic seemed to keep current state.
                            # Let's keep current state if it's not worse, to allow further moves.
                            # But if it's worse, revert.
                            if len(active_blocks) > best_size:
                                # Revert
                                active_blocks = set(best_blocks_copy)
                                cover_count = best_count_copy[:]
                                # Revert the specific swap? 
                                # Actually, if we reverted to best, we should break out of this B's processing?
                                # Or just continue?
                                # Let's just revert and continue.
                                pass
            else:
                # No unique subsets, but remove failed? 
                # This implies cover_count[t] > 1 for all t in B.
                # So B is not uniquely covering anything.
                # We can try to swap with a block that covers *more* unique items?
                # Or just skip.
                pass
        
        if not improved:
            # Try perturbation: Add a random block, then minimize.
            # If size decreases or stays same, keep.
            inactive_blocks = [b for b in range(num_k_subsets) if b not in active_blocks]
            if inactive_blocks:
                B_add = random.choice(inactive_blocks)
                active_blocks.add(B_add)
                for t_idx in block_t_indices[B_add]:
                    cover_count[t_idx] += 1
                active_blocks, cover_count = minimize(active_blocks, cover_count)
                
                if len(active_blocks) <= best_size:
                    if len(active_blocks) < best_size:
                        best_size = len(active_blocks)
                        best_blocks_copy = set(active_blocks)
                        best_count_copy = list(cover_count)
                else:
                    # Revert
                    active_blocks = set(best_blocks_copy)
                    cover_count = best_count_copy[:]

    # Final cleanup: minimize best copy
    best_blocks_copy, best_count_copy = minimize(best_blocks_copy, best_count_copy)
    
    result = [list(all_k_subsets[b_idx]) for b_idx in best_blocks_copy]
    return result
