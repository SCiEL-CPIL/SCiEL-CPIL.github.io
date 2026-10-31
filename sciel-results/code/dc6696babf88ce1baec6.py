import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {tuple(s): i for i, s in enumerate(t_subsets)}
    num_tsubs = len(t_subsets)
    
    # Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    # Precompute for each block, which t-subsets it covers
    block_to_tsubs = []
    for block in k_subsets:
        covered_tsubs = []
        for tsub in itertools.combinations(block, t):
            idx = t_subsets_set[tsub]
            covered_tsubs.append(idx)
        block_to_tsubs.append(covered_tsubs)
        
    # Precompute for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tsubs in enumerate(block_to_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
            
    # Convert to numpy arrays for fast membership testing
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    # Also create a set version for O(1) lookup
    tsub_to_blocks_set = [set(lst) for lst in tsub_to_blocks]
    
    # Initial gains: each block covers C(k, t) t-subsets
    k_choose_t = len(block_to_tsubs[0])
    gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
    
    # Track uncovered t-subsets
    uncovered = np.ones(num_tsubs, dtype=bool)
    
    selected_block_indices = []
    
    # 1. Greedy Construction
    while uncovered.any():
        if time.time() > deadline - 30:  # Leave 30s for local search
            break
            
        # Find the block with the maximum gain
        b = np.argmax(gains)
        
        if gains[b] == 0:
            break
            
        selected_block_indices.append(b)
        
        # Update gains for all blocks that share any t-subset with b
        for s in block_to_tsubs[b]:
            if uncovered[s]:
                uncovered[s] = False
                gains[tsub_to_blocks_np[s]] -= 1
    
    # If greedy didn't finish, complete it quickly
    while uncovered.any():
        if time.time() > deadline - 10:
            break
        # Find block covering most uncovered
        best_b = -1
        best_gain = 0
        for b in range(num_blocks):
            g = 0
            for s in block_to_tsubs[b]:
                if uncovered[s]:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_b = b
        if best_b < 0 or best_gain == 0:
            break
        selected_block_indices.append(best_b)
        for s in block_to_tsubs[best_b]:
            if uncovered[s]:
                uncovered[s] = False
    
    if not uncovered.any():
        # 2. Local Optimization
        # Maintain cover_count for each t-subset
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_block_indices:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        def is_redundant(b_idx):
            """Check if block b_idx is redundant (all its t-subsets covered by others)"""
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        def find_swap_in(tsubs_to_cover, exclude_set):
            """Find a block not in exclude_set that covers all tsubs in tsubs_to_cover"""
            if not tsubs_to_cover:
                return None
            first_s = tsubs_to_cover[0]
            candidates = tsub_to_blocks_set[first_s]
            for c in candidates:
                if c in exclude_set:
                    continue
                covers_all = True
                for s in tsubs_to_cover[1:]:
                    if c not in tsub_to_blocks_set[s]:
                        covers_all = False
                        break
                if covers_all:
                    return c
            return None
        
        # Local Search Loop with time budget
        local_deadline = deadline - 5
        improved = True
        while improved and time.time() < local_deadline:
            improved = False
            
            # Step 2a: Redundancy Removal - check ALL blocks
            redundant_blocks = []
            for b in selected_block_indices:
                if is_redundant(b):
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                for b in redundant_blocks:
                    if b in selected_block_indices:
                        selected_block_indices.remove(b)
                        for s in block_to_tsubs[b]:
                            cover_count[s] -= 1
                improved = True
                continue  # Restart to find new redundancies
            
            # Step 2b: Try Swaps - focus on blocks with fewest unique t-subsets
            unique_counts = {}
            unique_tsubs = {}
            for b in selected_block_indices:
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_list.append(s)
                unique_counts[b] = len(u_list)
                unique_tsubs[b] = u_list
            
            # Sort by unique count (ascending) - try to remove blocks with fewest uniques first
            sorted_blocks = sorted(selected_block_indices, key=lambda b: unique_counts[b])
            
            swap_made = False
            for b_out in sorted_blocks:
                if time.time() > local_deadline:
                    break
                s_uniq = unique_tsubs[b_out]
                
                # Try to find a swap-in block
                exclude_set = set(selected_block_indices)
                b_in = find_swap_in(s_uniq, exclude_set)
                
                if b_in is not None:
                    # Perform swap
                    selected_block_indices.remove(b_out)
                    selected_block_indices.append(b_in)
                    
                    # Update cover counts
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    
                    swap_made = True
                    improved = True
                    break  # Restart loop after one swap
            
            if not swap_made:
                # Try 2-for-1: find two blocks whose unique t-subsets are all covered by a third block
                if time.time() > local_deadline - 10:
                    break
                
                # For each pair of blocks, check if their union of unique t-subsets is covered by some other block
                n_sel = len(selected_block_indices)
                found_2for1 = False
                for i in range(n_sel):
                    if time.time() > local_deadline:
                        break
                    b1 = selected_block_indices[i]
                    u1 = unique_tsubs[b1]
                    if len(u1) > 5:  # Skip if too many uniques
                        continue
                    for j in range(i + 1, n_sel):
                        if time.time() > local_deadline:
                            break
                        b2 = selected_block_indices[j]
                        u2 = unique_tsubs[b2]
                        if len(u2) > 5:
                            continue
                        
                        # Union of unique t-subsets
                        u_union = set(u1) | set(u2)
                        if len(u_union) > 10:
                            continue
                        
                        # Find a block that covers all of u_union
                        exclude_set = set(selected_block_indices)
                        b3 = find_swap_in(list(u_union), exclude_set)
                        
                        if b3 is not None:
                            # Remove b1 and b2, add b3
                            selected_block_indices.remove(b1)
                            selected_block_indices.remove(b2)
                            selected_block_indices.append(b3)
                            
                            # Update cover counts
                            for s in block_to_tsubs[b1]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[b2]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[b3]:
                                cover_count[s] += 1
                            
                            found_2for1 = True
                            improved = True
                            break
                    if found_2for1:
                        break
                
                if not found_2for1:
                    break  # Local minimum, stop

    # Convert selected block indices back to the actual subsets
    result = [list(k_subsets[i]) for i in selected_block_indices]
    
    # Deduplicate (shouldn't be needed but safety)
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
