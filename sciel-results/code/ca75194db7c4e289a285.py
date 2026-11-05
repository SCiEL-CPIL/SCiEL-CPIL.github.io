import itertools
import time
import random
import numpy as np
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s kill, 10s buffer)

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # Build incidence structure
    block_t_indices = []
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]

    for b_idx, block in enumerate(all_k_blocks):
        t_subs_in_block = list(itertools.combinations(block, t))
        indices = [t_subset_to_idx[s] for s in t_subs_in_block]
        block_t_indices.append(indices)
        for idx in indices:
            t_subset_to_blocks[idx].append(b_idx)

    # Convert block_t_indices to numpy arrays for fast access
    block_t_indices_arr = [np.array(indices, dtype=np.int32) for indices in block_t_indices]

    # Build sparse incidence matrix
    row_data = []
    col_data = []
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            row_data.append(b_idx)
            col_data.append(idx)

    incidence = csr_matrix(
        (np.ones(len(row_data), dtype=np.int32), (row_data, col_data)),
        shape=(num_k_blocks, num_t_subsets)
    )

    # Precompute block sizes (number of t-subsets per block) - they're all C(k,t)
    block_size = len(block_t_indices[0])

    def greedy_construct(seed, deadline_check=True):
        """Vectorized greedy construction."""
        rng = random.Random(seed)
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        selected_blocks = []
        selected_set = set()
        
        uncovered_count = num_t_subsets
        all_indices = np.arange(num_k_blocks)
        
        # Track which blocks are still available
        available = np.ones(num_k_blocks, dtype=bool)
        
        while uncovered_count > 0:
            if deadline_check and time.time() > deadline - 25.0:
                break
                
            # Compute gain for all available blocks in parallel
            # gain[b] = number of uncovered t-subsets in block b
            # We can compute this by: for each available block, count zeros in coverage over its t-subsets
            # This is expensive to do fully vectorized, so we use a smarter approach:
            # gain[b] = block_size - sum(coverage > 0 over block's t-subsets)
            # Actually: gain = number of t-subsets in block where coverage == 0
            
            # For speed, sample a subset of available blocks if too many
            avail_count = np.sum(available)
            if avail_count == 0:
                break
            
            # If many available, sample; otherwise evaluate all
            if avail_count > 5000:
                avail_list = np.where(available)[0]
                # Sample up to 5000
                sample_size = min(5000, avail_count)
                sampled = rng.sample(avail_list.tolist(), sample_size)
                eval_indices = np.array(sampled, dtype=np.int32)
            else:
                eval_indices = np.where(available)[0]
            
            # Compute gains for evaluated blocks
            gains = np.zeros(len(eval_indices), dtype=np.int32)
            for i, b_idx in enumerate(eval_indices):
                gains[i] = np.sum(coverage[block_t_indices_arr[b_idx]] == 0)
            
            if len(gains) == 0:
                break
                
            best_local = np.argmax(gains)
            best_gain = gains[best_local]
            if best_gain == 0:
                break
                
            best_block = eval_indices[best_local]
            selected_blocks.append(best_block)
            selected_set.add(best_block)
            coverage[block_t_indices_arr[best_block]] += 1
            uncovered_count -= best_gain
            available[best_block] = False
            
        return selected_blocks

    def reduce_covering(blocks, deadline_check=True):
        """Remove redundant blocks iteratively."""
        blocks = set(blocks)
        if not blocks:
            return blocks
            
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in blocks:
            coverage[block_t_indices_arr[b_idx]] += 1
            
        changed = True
        while changed:
            changed = False
            to_remove = []
            for b_idx in blocks:
                if np.any(coverage[block_t_indices_arr[b_idx]] == 1):
                    continue
                else:
                    to_remove.append(b_idx)
                    
            for b_idx in to_remove:
                blocks.remove(b_idx)
                coverage[block_t_indices_arr[b_idx]] -= 1
                changed = True
                
            if deadline_check and time.time() > deadline - 10.0:
                break
                
        return blocks

    def local_search(blocks, seed, deadline_check=True):
        """Local search: swap-based and deletion-based improvements."""
        rng = random.Random(seed)
        blocks = set(blocks)
        
        if not blocks:
            return blocks
            
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in blocks:
            coverage[block_t_indices_arr[b_idx]] += 1
        
        initial_size = len(blocks)
        
        # Phase 1: Deletion pass
        changed = True
        while changed and (not deadline_check or time.time() < deadline - 15.0):
            changed = False
            for b_idx in list(blocks):
                if np.all(coverage[block_t_indices_arr[b_idx]] > 1):
                    blocks.remove(b_idx)
                    coverage[block_t_indices_arr[b_idx]] -= 1
                    changed = True
        
        # Phase 2: Swap-based local search
        steps = 0
        while (not deadline_check or time.time() < deadline - 10.0) and blocks:
            steps += 1
            
            # Pick a random block to try to swap
            b_idx = rng.choice(list(blocks))
            
            # Find unique t-subsets covered by this block
            unique_mask = (coverage[block_t_indices_arr[b_idx]] == 1)
            if not np.any(unique_mask):
                # Can delete directly
                blocks.remove(b_idx)
                coverage[block_t_indices_arr[b_idx]] -= 1
                continue
                
            unique_indices = block_t_indices_arr[b_idx][unique_mask]
            
            # Find candidate replacement blocks that cover all unique t-subsets
            # Use intersection of t_subset_to_blocks for each unique subset
            if len(unique_indices) > 6:
                # Too many unique constraints, skip
                continue
                
            # Start with smallest constraint
            min_set_idx = min(unique_indices, key=lambda i: len(t_subset_to_blocks[i]))
            candidates = set(t_subset_to_blocks[min_set_idx])
            
            for idx in unique_indices:
                if idx != min_set_idx:
                    candidates.intersection_update(t_subset_to_blocks[idx])
            
            candidates -= blocks
            
            if not candidates:
                continue
                
            # Try random candidates
            for attempt in range(3):
                b_new = rng.choice(list(candidates))
                
                # Temporarily add
                blocks.add(b_new)
                coverage[block_t_indices_arr[b_new]] += 1
                
                # Try to remove b_idx
                removed_count = 0
                if np.all(coverage[block_t_indices_arr[b_idx]] > 1):
                    blocks.remove(b_idx)
                    coverage[block_t_indices_arr[b_idx]] -= 1
                    removed_count += 1
                    
                # Try to remove other redundant blocks
                for b_rem in list(blocks):
                    if np.all(coverage[block_t_indices_arr[b_rem]] > 1):
                        blocks.remove(b_rem)
                        coverage[block_t_indices_arr[b_rem]] -= 1
                        removed_count += 1
                
                # If we didn't improve size, revert
                if removed_count == 0:
                    blocks.remove(b_new)
                    coverage[block_t_indices_arr[b_new]] -= 1
                elif removed_count < 2:
                    # Size same or worse, revert if not improved
                    if removed_count <= 1:
                        blocks.remove(b_new)
                        coverage[block_t_indices_arr[b_new]] -= 1
                        # Re-add b_idx if we removed it
                        if b_idx not in blocks:
                            blocks.add(b_idx)
                            coverage[block_t_indices_arr[b_idx]] += 1
                        # Re-add other removed blocks - this is complex, so just skip
                        # Actually, let's just keep the state if size improved
                        if len(blocks) < initial_size or len(blocks) < len(blocks) + removed_count:
                            pass  # Keep improvement
                        else:
                            # Complex revert, just break
                            break

        return blocks

    def is_valid(blocks):
        """Check if blocks form a valid covering."""
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in blocks:
            coverage[block_t_indices_arr[b_idx]] += 1
        return np.all(coverage > 0)

    # Main execution
    best_blocks = None
    best_count = float('inf')
    
    # 1. Multi-restart Greedy + Reduction
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline - 40.0:
            break
            
        seed = i * 7919 + 12345
        blocks = greedy_construct(seed)
        
        if not blocks:
            continue
            
        if not is_valid(blocks):
            continue
            
        # Reduce
        reduced_blocks = reduce_covering(blocks)
        
        if len(reduced_blocks) < best_count:
            best_count = len(reduced_blocks)
            best_blocks = set(reduced_blocks)
            if best_count <= 300:
                break

    if best_blocks is None:
        # Fallback: full greedy
        blocks = greedy_construct(0)
        if blocks:
            best_blocks = set(blocks)
        else:
            return []

    # 2. Local Search on the best solution
    if time.time() < deadline - 15.0:
        final_blocks = local_search(set(best_blocks), seed=999)
        if is_valid(final_blocks):
            if len(final_blocks) < best_count:
                best_blocks = final_blocks
                best_count = len(final_blocks)
    
    # 3. Additional local search rounds with different seeds
    for ls_seed in [123, 456, 789]:
        if time.time() > deadline - 10.0:
            break
        candidate = local_search(set(best_blocks), seed=ls_seed)
        if is_valid(candidate) and len(candidate) < best_count:
            best_blocks = candidate
            best_count = len(candidate)

    return [list(all_k_blocks[i]) for i in sorted(best_blocks)]
