import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    TIME_LIMIT = 115.0  # safety margin before 150s kill
    
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

    # 2. Greedy Set Cover
    uncovered = [True] * num_t_subsets
    gain = [0] * num_k_subsets
    
    for b_idx in range(num_k_subsets):
        gain[b_idx] = len(block_t_indices[b_idx])
        
    selected_blocks = []
    
    while True:
        max_gain = max(gain)
        if max_gain == 0:
            break
        best_block = gain.index(max_gain)
        selected_blocks.append(best_block)
        
        for t_idx in block_t_indices[best_block]:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                for b_idx in t_subset_blocks[t_idx]:
                    gain[b_idx] -= 1
                    
    # 3. Minimalization Helper
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

    # Initialize active set and counts
    active_blocks = set(selected_blocks)
    cover_count = [0] * num_t_subsets
    for b_idx in active_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    # Initial minimalization
    active_blocks, cover_count = minimize(active_blocks, cover_count)
    
    best_count = len(active_blocks)
    best_blocks = set(active_blocks)
    best_cover = list(cover_count)

    # 4. Local Search (Hill Climbing with Swaps)
    while time.time() - start_time < TIME_LIMIT:
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)
        
        for B in current_blocks:
            if time.time() - start_time > TIME_LIMIT:
                break
            if B not in active_blocks:
                continue
            
            # Find t-subsets uniquely covered by B
            uncovered_by_B = []
            for t_idx in block_t_indices[B]:
                if cover_count[t_idx] == 1:
                    uncovered_by_B.append(t_idx)
            
            if not uncovered_by_B:
                continue

            # Sort by number of covering blocks to speed up intersection
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
            
            if not candidates:
                continue
            
            # Filter candidates
            valid_candidates = []
            for c in candidates:
                if c != B and c not in active_blocks:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                continue
            
            # Perform Swap
            B_new = random.choice(valid_candidates)
            
            active_blocks.remove(B)
            active_blocks.add(B_new)
            
            for t_idx in block_t_indices[B]:
                cover_count[t_idx] -= 1
            for t_idx in block_t_indices[B_new]:
                cover_count[t_idx] += 1
            
            # Minimize after swap
            active_blocks, cover_count = minimize(active_blocks, cover_count)
            improved = True
            
            if len(active_blocks) < best_count:
                best_count = len(active_blocks)
                best_blocks = set(active_blocks)
                best_cover = list(cover_count)

        if not improved:
            # Try add-remove moves to escape local minima
            add_improved = False
            num_adds = 0
            max_adds = 5  # limit to keep time bounded
            
            for _ in range(max_adds):
                if time.time() - start_time > TIME_LIMIT:
                    break
                if num_adds >= max_adds:
                    break
                    
                # Pick a random uncovered t-subset to target
                # First find any t-subset with minimum coverage
                min_cov = min(cover_count)
                if min_cov >= 2:
                    break
                    
                target_t = random.choice([i for i, c in enumerate(cover_count) if c == min_cov])
                
                # Pick a random block that covers this target
                candidate_blocks = t_subset_blocks[target_t]
                if not candidate_blocks:
                    break
                    
                B_add = random.choice(candidate_blocks)
                if B_add in active_blocks:
                    continue
                    
                active_blocks.add(B_add)
                for t_idx in block_t_indices[B_add]:
                    cover_count[t_idx] += 1
                num_adds += 1
                add_improved = True
                
                # Now try to remove blocks
                active_blocks, cover_count = minimize(active_blocks, cover_count)
                
                if len(active_blocks) < best_count:
                    best_count = len(active_blocks)
                    best_blocks = set(active_blocks)
                    best_cover = list(cover_count)
                    
                # Reset for next add
                if add_improved:
                    # We added one, now see if we can remove more than one
                    # Actually minimize already did that
                    pass

            if not improved and not add_improved:
                break

    # Return best result found
    result = [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
    return result
