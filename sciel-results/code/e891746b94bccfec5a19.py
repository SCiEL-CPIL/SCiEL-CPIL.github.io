import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    # Generate all t-subsets and k-blocks
    all_t_subsets = list(itertools.combinations(range(v), t))
    all_k_blocks = list(itertools.combinations(range(v), k))
    
    # Map each t-subset to its index for quick lookup
    t_subset_index = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Map each k-block to the list of t-subsets it covers (as indices)
    block_to_t_indices = []
    for block in all_k_blocks:
        indices = [t_subset_index[sub] for sub in itertools.combinations(block, t)]
        block_to_t_indices.append(indices)
        
    # Map each t-subset index to the list of k-block indices that cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, indices in enumerate(block_to_t_indices):
        for t_idx in indices:
            t_to_blocks[t_idx].append(b_idx)

    # Helper: Get t-subsets covered by a block index
    # We can use block_to_t_indices directly.

    def greedy_random(seed):
        random.seed(seed)
        uncovered = set(range(num_t))
        cover = []
        
        while uncovered:
            # Pick a random uncovered t-subset
            s_idx = random.choice(list(uncovered))
            # Pick a random block that covers it
            b_idx = random.choice(t_to_blocks[s_idx])
            
            cover.append(b_idx)
            
            # Remove all t-subsets covered by this block from uncovered
            for t_idx in block_to_t_indices[b_idx]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    
        return cover

    def local_search_delete(cover):
        # Compute coverage counts for each t-subset
        counts = [0] * num_t
        for b_idx in cover:
            for t_idx in block_to_t_indices[b_idx]:
                counts[t_idx] += 1
                
        improved = True
        # Limit passes to avoid infinite loops or excessive time
        max_passes = 50
        
        while improved and max_passes > 0:
            improved = False
            max_passes -= 1
            
            # Shuffle cover to avoid bias
            shuffled_cover = cover.copy()
            random.shuffle(shuffled_cover)
            
            for b_idx in shuffled_cover:
                if b_idx not in cover:
                    continue
                
                # Check if any t-subset covered by b_idx has count == 1
                is_essential = False
                for t_idx in block_to_t_indices[b_idx]:
                    if counts[t_idx] == 1:
                        is_essential = True
                        break
                
                if not is_essential:
                    # Block is redundant, remove it
                    cover.remove(b_idx)
                    for t_idx in block_to_t_indices[b_idx]:
                        counts[t_idx] -= 1
                    improved = True
                    # Restart the pass to check for new redundancies
                    break
                    
        return cover

    best_cover = None
    best_len = float('inf')
    seed_counter = 0
    
    while time.time() < deadline - 5.0:
        seed = seed_counter
        seed_counter += 1
        
        # Run randomized greedy
        cover = greedy_random(seed)
        
        # Apply deletion local search
        cover = local_search_delete(cover)
        
        # Check validity (should be valid by construction, but safety check)
        # For speed, we trust the construction.
        
        if len(cover) < best_len:
            best_len = len(cover)
            best_cover = cover.copy()
            
        # Early exit if we hit a very good bound (optional)
        if best_len <= 300:
            break

    if best_cover is None:
        # Fallback
        best_cover = greedy_random(0)
        best_cover = local_search_delete(best_cover)

    # Convert indices back to point lists
    result = [list(all_k_blocks[b_idx]) for b_idx in best_cover]
    
    return result
