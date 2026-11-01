import itertools
import random
import time

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
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

    # 4. Local Search (Hill Climbing with Swaps)
    start_time = time.time()
    time_limit = 110.0
    
    while time.time() - start_time < time_limit:
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)
        
        for B in current_blocks:
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
            
        if not improved:
            break

    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    return result
