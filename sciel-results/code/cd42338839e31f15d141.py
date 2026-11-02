import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    total_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute list of subset indices for each block
    block_subset_indices = []
    for b in all_blocks:
        subs = itertools.combinations(b, t)
        indices = [subset_to_idx[s] for s in subs]
        block_subset_indices.append(indices)
        
    # Precompute bitsets for each block
    block_bitsets = []
    for indices in block_subset_indices:
        bits = 0
        for idx in indices:
            bits |= (1 << idx)
        block_bitsets.append(bits)
            
    mask_all = (1 << total_subsets) - 1
    
    def greedy_randomized_construction(top_k=10):
        uncovered_bits = mask_all
        current_blocks = []
        used_blocks = [False] * num_blocks
        
        while uncovered_bits != 0:
            best_gains = []
            for bi in range(num_blocks):
                if used_blocks[bi]:
                    continue
                gain = (uncovered_bits & block_bitsets[bi]).bit_count()
                if gain > 0:
                    best_gains.append((gain, bi))
            
            if not best_gains:
                break
                
            # Sort by gain descending
            best_gains.sort(key=lambda x: x[0], reverse=True)
            
            # Pick from top K
            k_val = min(top_k, len(best_gains))
            chosen = random.choice(best_gains[:k_val])
            bi = chosen[1]
            
            current_blocks.append(bi)
            used_blocks[bi] = True
            uncovered_bits &= ~block_bitsets[bi]
            
        return current_blocks

    best_solution = None
    start_time = time.time()
    time_limit = 115
    
    # Run GRASP iterations
    while time.time() - start_time < time_limit:
        sol = greedy_randomized_construction(top_k=10)
        
        # Local Search
        current_blocks = list(sol)
        current_set = set(current_blocks)
        
        counts = np.zeros(total_subsets, dtype=np.int32)
        for bi in current_blocks:
            for idx in block_subset_indices[bi]:
                counts[idx] += 1
                
        # Phase 1: Removal
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                bi = current_blocks[i]
                redundant = True
                for idx in block_subset_indices[bi]:
                    if counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    current_blocks.pop(i)
                    current_set.remove(bi)
                    for idx in block_subset_indices[bi]:
                        counts[idx] -= 1
                    changed = True
                    
        # Phase 2: Swap
        # Try to swap blocks to enable more removals
        for _ in range(2): # Limit passes
            improved = True
            while improved:
                improved = False
                indices_to_check = list(range(len(current_blocks)))
                random.shuffle(indices_to_check)
                
                for i in indices_to_check:
                    if i >= len(current_blocks): continue
                    bi = current_blocks[i]
                    
                    # Get unique subsets
                    unique_mask = 0
                    for idx in block_subset_indices[bi]:
                        if counts[idx] == 1:
                            unique_mask |= (1 << idx)
                            
                    if unique_mask == 0:
                        # Redundant
                        current_blocks.pop(i)
                        current_set.remove(bi)
                        for idx in block_subset_indices[bi]:
                            counts[idx] -= 1
                        improved = True
                        break
                    
                    # Find a block bj that covers unique_mask
                    best_bj = -1
                    # Iterate through all blocks, check bitset intersection
                    for bj in range(num_blocks):
                        if bj in current_set:
                            continue
                        if (block_bitsets[bj] & unique_mask) == unique_mask:
                            best_bj = bj
                            break
                            
                    if best_bj != -1:
                        # Perform swap
                        current_blocks[i] = best_bj
                        current_set.remove(bi)
                        current_set.add(best_bj)
                        
                        # Update counts
                        for idx in block_subset_indices[bi]:
                            counts[idx] -= 1
                        for idx in block_subset_indices[best_bj]:
                            counts[idx] += 1
                            
                        improved = True
                        break # Restart loop as counts changed
        
        if best_solution is None or len(current_blocks) < len(best_solution):
            best_solution = list(current_blocks)
            
        if time.time() - start_time > time_limit:
            break
            
    if best_solution is None:
        # Fallback to greedy
        best_solution = greedy_randomized_construction(top_k=1)
        
    return [list(all_blocks[bi]) for bi in best_solution]
