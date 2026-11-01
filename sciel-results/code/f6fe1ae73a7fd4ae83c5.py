import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    # 2. Phase 1: Greedy Packing
    pool = set(range(num_k_subsets))
    core_blocks = []
    uncovered_mask = [1] * num_t_subsets
    
    while pool:
        best_block = -1
        max_cov = -1
        
        for b_idx in pool:
            cov = 0
            for t_idx in block_t_indices[b_idx]:
                if uncovered_mask[t_idx]:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
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
        # Use a boolean array for O(1) lookup instead of set
        is_uncovered = [False] * num_t_subsets
        for i in still_uncovered:
            is_uncovered[i] = True
        
        while True:
            # Check if any uncovered remain
            any_uncovered = False
            for i in range(num_t_subsets):
                if is_uncovered[i]:
                    any_uncovered = True
                    break
            if not any_uncovered:
                break
            
            if time.time() > deadline - 10.0:
                break
            
            best_block = -1
            max_gain = 0
            
            for b_idx in range(num_k_subsets):
                if b_idx in active_blocks:
                    continue
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if is_uncovered[t_idx]:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
                    
            if best_block == -1 or max_gain == 0:
                break
                
            active_blocks.add(best_block)
            for t_idx in block_t_indices[best_block]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                cover_count[t_idx] += 1

    # 4. Minimalization
    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
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

    # 5. Local Search (Hill Climbing with Swaps, Adds, Removes)
    while time.time() < deadline - 5.0:
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)
        
        for B in current_blocks:
            if B not in active_blocks:
                continue
            
            # Try swap: find blocks that can replace B
            uncovered_by_B = []
            for t_idx in block_t_indices[B]:
                if cover_count[t_idx] == 1:
                    uncovered_by_B.append(t_idx)
            
            if uncovered_by_B:
                uncovered_by_B.sort(key=lambda x: len(t_subset_blocks[x]))
                
                candidates = None
                for t_idx in uncovered_by_B:
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
                        B_new = random.choice(valid_candidates)
                        
                        active_blocks.remove(B)
                        active_blocks.add(B_new)
                        
                        for t_idx in block_t_indices[B]:
                            cover_count[t_idx] -= 1
                        for t_idx in block_t_indices[B_new]:
                            cover_count[t_idx] += 1
                        
                        improved = True
                        # Don't minimize here; do it in batch at end of pass
                        continue
            
            # Try remove: if B is redundant, remove it
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
                continue
            
            # Try add: find a block that covers the most uncovered t-subsets
            # Only try this occasionally to avoid overhead
            if random.random() < 0.1:
                # Find t-subsets with minimum coverage
                min_cover = min(cover_count)
                if min_cover > 0:
                    # Find blocks that could improve coverage
                    best_add = -1
                    best_gain = 0
                    for b_idx in range(num_k_subsets):
                        if b_idx in active_blocks:
                            continue
                        gain = 0
                        for t_idx in block_t_indices[b_idx]:
                            if cover_count[t_idx] == min_cover:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_add = b_idx
                    if best_add != -1 and best_gain > 0:
                        active_blocks.add(best_add)
                        for t_idx in block_t_indices[best_add]:
                            cover_count[t_idx] += 1
                        improved = True
        
        # Batch minimize at end of each pass
        if improved:
            active_blocks, cover_count = minimize(active_blocks, cover_count)
        else:
            break

    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    return result
