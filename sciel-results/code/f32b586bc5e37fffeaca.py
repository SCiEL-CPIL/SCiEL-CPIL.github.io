import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each k-subset, compute the set of t-subset indices it covers (as a frozenset for hashing, and as a sorted list for iteration)
    block_t_indices = []
    for block in all_k_subsets:
        indices = tuple(sorted(t_subset_to_idx[s] for s in itertools.combinations(block, t)))
        block_t_indices.append(indices)
    
    # For each t-subset, which blocks cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
    
    # Helper: minimize a given set of blocks (remove redundant blocks)
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
    
    # Helper: Greedy Set Cover with randomization among ties
    def greedy_cover():
        uncovered = [True] * num_t_subsets
        gain = [len(block_t_indices[i]) for i in range(num_k_subsets)]
        selected_blocks = []
        
        while True:
            # Find max gain
            max_gain = max(gain)
            if max_gain == 0:
                break
            
            # Find all blocks with max gain, pick randomly
            candidates = [i for i in range(num_k_subsets) if gain[i] == max_gain]
            best_block = random.choice(candidates)
            
            selected_blocks.append(best_block)
            
            # Update gains
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_subset_blocks[t_idx]:
                        gain[b_idx] -= 1
                        
        return selected_blocks
    
    # Helper: Local Search with swap + minimize
    def local_search(active_blocks, cover_count, time_limit):
        start_time_ls = time.time()
        
        while time.time() - start_time_ls < time_limit:
            improved = False
            current_blocks = list(active_blocks)
            random.shuffle(current_blocks)
            
            for B in current_blocks:
                if time.time() - start_time_ls >= time_limit:
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

                # Find candidate blocks that cover all of uncovered_by_B
                # Start with the smallest t_subset_blocks list
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
                
                # Filter: must not be B, must not already be in active_blocks
                valid_candidates = [c for c in candidates if c != B and c not in active_blocks]
                
                if not valid_candidates:
                    continue
                
                # Perform swap
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
    
    # Main Loop: Multi-Start
    best_blocks = None
    best_size = float('inf')
    
    # First run: quick greedy to get a baseline
    selected = greedy_cover()
    active_blocks = set(selected)
    cover_count = [0] * num_t_subsets
    for b_idx in active_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
    active_blocks, cover_count = minimize(active_blocks, cover_count)
    best_size = len(active_blocks)
    best_blocks = list(active_blocks)
    
    # Continue with more iterations if time allows
    while time.time() < deadline - 2.0:
        # Greedy with randomization
        selected = greedy_cover()
        
        # Minimize
        active_blocks = set(selected)
        cover_count = [0] * num_t_subsets
        for b_idx in active_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
        active_blocks, cover_count = minimize(active_blocks, cover_count)
        
        # Local Search with remaining time budget
        remaining_time = deadline - 2.0 - time.time()
        if remaining_time > 0.5:
            # Give local search half the remaining time, at least 0.2s
            ls_time = max(0.2, remaining_time * 0.4)
            active_blocks, cover_count = local_search(active_blocks, cover_count, ls_time)
        
        # Update Best
        current_size = len(active_blocks)
        if current_size < best_size:
            best_size = current_size
            best_blocks = list(active_blocks)
            
            # Lower bound is 131; if we hit it, stop
            if best_size <= 131:
                break
    
    if best_blocks is None:
        selected = greedy_cover()
        best_blocks = selected

    result = [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
    return result
