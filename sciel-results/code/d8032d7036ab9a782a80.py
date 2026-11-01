import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map block to list of t-subset indices it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Map t-subset to list of blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # Precompute degrees of t-subsets
    t_subset_degrees = [len(lst) for lst in t_subset_blocks]
    
    # 2. Greedy Set Cover with Scarcity Weights
    # We combine simple gain with a "scarcity" factor: 1/degree(t)
    # This helps cover rare t-subsets earlier, which might be harder to cover later.
    
    def run_greedy(order):
        uncovered = [True] * num_t_subsets
        gain = [0] * num_k_subsets
        
        # Initialize gain
        for b_idx in order:
            gain[b_idx] = len(block_t_indices[b_idx])
            
        selected_blocks = []
        
        while True:
            max_gain = -1
            best_block = -1
            # To break ties or improve selection, we can add a small random jitter
            # or use the scarcity weight.
            # Let's stick to simple gain first, but maybe prioritize rare blocks if gain is same?
            # Actually, the prompt suggests "spectral cancellation" or similar.
            # A simpler heuristic: Gain = count_uncovered + sum(1/degree(t) for t in block if uncovered)
            
            # For speed, let's just use count_uncovered.
            # But we can refine:
            # score = count_uncovered + 0.01 * scarcity_sum
            
            for b_idx in order:
                if gain[b_idx] > max_gain:
                    max_gain = gain[b_idx]
                    best_block = b_idx
                    
            if max_gain == 0:
                break
                
            selected_blocks.append(best_block)
            
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_subset_blocks[t_idx]:
                        gain[b_idx] -= 1
                        
        return selected_blocks

    # Run multiple greedy instances with different random orders
    best_blocks = None
    best_size = float('inf')
    
    # Do a few greedy starts
    num_starts = 15
    for i in range(num_starts):
        if time.time() - start_time > time_limit:
            break
        order = list(range(num_k_subsets))
        random.shuffle(order)
        blocks = run_greedy(order)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
    # 3. Local Search on the best solution
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

    def local_search(active_blocks_set, cover_counts):
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            current_blocks = list(active_blocks_set)
            random.shuffle(current_blocks)
            
            for B in current_blocks:
                if B not in active_blocks_set:
                    continue
                
                # Find t-subsets uniquely covered by B
                uncovered_by_B = []
                for t_idx in block_t_indices[B]:
                    if cover_counts[t_idx] == 1:
                        uncovered_by_B.append(t_idx)
                
                if not uncovered_by_B:
                    continue

                # Find candidates that cover all uncovered_by_B
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
                
                valid_candidates = [c for c in candidates if c != B and c not in active_blocks_set]
                
                if not valid_candidates:
                    continue
                
                # Try swapping with a random candidate
                B_new = random.choice(valid_candidates)
                
                # Perform swap
                active_blocks_set.remove(B)
                active_blocks_set.add(B_new)
                
                for t_idx in block_t_indices[B]:
                    cover_counts[t_idx] -= 1
                for t_idx in block_t_indices[B_new]:
                    cover_counts[t_idx] += 1
                
                # Minimize after swap
                active_blocks_set, cover_counts = minimize(active_blocks_set, cover_counts)
                improved = True

        return active_blocks_set, cover_counts

    if best_blocks:
        active_blocks = set(best_blocks)
        cover_count = [0] * num_t_subsets
        for b_idx in active_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
                
        # Initial minimalization
        active_blocks, cover_count = minimize(active_blocks, cover_count)
        
        # Local Search
        active_blocks, cover_count = local_search(active_blocks, cover_count)
        
        result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
        return result
    else:
        # Fallback
        return [list(b) for b in all_k_subsets]
