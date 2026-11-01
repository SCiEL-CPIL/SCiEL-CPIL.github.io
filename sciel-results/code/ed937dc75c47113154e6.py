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

    def greedy_set_cover():
        uncovered = [True] * num_t_subsets
        gain = [len(block_t_indices[b]) for b in range(num_k_subsets)]
        selected_blocks = []
        
        while True:
            max_gain = max(gain)
            if max_gain == 0:
                break
            
            # Find all blocks with max gain, pick one randomly
            candidates = [i for i, g in enumerate(gain) if g == max_gain]
            best_block = random.choice(candidates)
            selected_blocks.append(best_block)
            
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_subset_blocks[t_idx]:
                        gain[b_idx] -= 1
                        
        return selected_blocks

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

    def local_search(active_blocks, cover_count, time_limit):
        start_time = time.time()
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
        return active_blocks, cover_count

    best_blocks = None
    best_size = float('inf')
    total_time_limit = 110.0
    start_total = time.time()
    
    while time.time() - start_total < total_time_limit:
        # 1. Greedy
        selected = greedy_set_cover()
        active_blocks = set(selected)
        cover_count = [0] * num_t_subsets
        for b_idx in active_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
        
        # 2. Minimize
        active_blocks, cover_count = minimize(active_blocks, cover_count)
        
        # 3. Local Search
        # Allocate remaining time / 5 for local search in this iteration
        remaining = total_time_limit - (time.time() - start_total)
        ls_time = max(1.0, remaining / 5.0)
        active_blocks, cover_count = local_search(active_blocks, cover_count, ls_time)
        
        if len(active_blocks) < best_size:
            best_size = len(active_blocks)
            best_blocks = list(active_blocks)
            
    if best_blocks is None:
        # Fallback: just return the first one found
        selected = greedy_set_cover()
        active_blocks = set(selected)
        cover_count = [0] * num_t_subsets
        for b_idx in active_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
        active_blocks, cover_count = minimize(active_blocks, cover_count)
        best_blocks = list(active_blocks)

    return [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
