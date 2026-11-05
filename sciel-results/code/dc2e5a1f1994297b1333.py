import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    # Precompute all k-blocks and their t-subset coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}

    # For each block, store the list of subset indices it covers
    block_to_subsets = [[] for _ in range(n_blocks)]
    # For each subset, store which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]

    for b_idx, block in enumerate(all_blocks):
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            block_to_subsets[b_idx].append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)

    # Initialize gain vector: gain[b] = number of currently uncovered subsets in block b
    uncovered = np.ones(n_subsets, dtype=bool)
    gains = np.zeros(n_blocks, dtype=np.int32)

    # Initial gains: for each subset, add 1 to all blocks containing it
    for u in range(n_subsets):
        for b in subset_to_blocks[u]:
            gains[b] += 1

    blocks_found = []
    used_blocks = set()

    # Helper to calculate gain of a specific block (tuple of points)
    def get_block_gain(block_tuple):
        b_idx = block_to_idx[block_tuple]
        count = 0
        for u in block_to_subsets[b_idx]:
            if uncovered[u]:
                count += 1
        return count

    # Helper to optimize a block using 1-swap local search
    def optimize_block(block, current_gain):
        best_block = list(block)
        best_gain = current_gain
        improved = True
        # Limit local search iterations to prevent excessive time usage
        max_iters = 10
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            # Try swapping each point in best_block with each point not in best_block
            for i, x in enumerate(best_block):
                for y in range(v):
                    if y == x:
                        continue
                    if y in best_block:
                        continue
                    
                    # Create new block
                    new_block = best_block[:]
                    new_block[i] = y
                    new_block_tuple = tuple(sorted(new_block))
                    
                    # Check if this block is valid (distinct elements) - guaranteed by construction
                    # Calculate gain
                    new_gain = get_block_gain(new_block_tuple)
                    
                    if new_gain > best_gain:
                        best_gain = new_gain
                        best_block = new_block
                        improved = True
        
        return tuple(sorted(best_block)), best_gain

    while True:
        if time.time() > deadline:
            break

        if not uncovered.any():
            break

        # Find block with max gain
        b_star = int(np.argmax(gains))

        if gains[b_star] <= 0:
            break

        # Get the best candidate block
        candidate_block = all_blocks[b_star]
        candidate_gain = gains[b_star]
        
        # Local optimization of this block
        # Note: This might be expensive. We only do it if we have time.
        # Let's limit local search depth or skip if time is tight.
        if time.time() < deadline - 5.0:
            optimized_block, optimized_gain = optimize_block(candidate_block, candidate_gain)
            if optimized_gain > candidate_gain:
                candidate_block = optimized_block
                candidate_gain = optimized_gain
        
        # Add this block
        block_tuple = tuple(candidate_block)
        
        # Verify it's not used (should be true if gain > 0)
        if block_tuple not in used_blocks:
            blocks_found.append(list(block_tuple))
            used_blocks.add(block_tuple)

            # Update uncovered and gains
            for subset in itertools.combinations(block_tuple, t):
                u = subset_to_idx[subset]
                if uncovered[u]:
                    uncovered[u] = False
                    # Decrement gain for all blocks containing u
                    for b in subset_to_blocks[u]:
                        gains[b] -= 1

            # Early exit if all covered
            if not uncovered.any():
                break
        else:
            # Fallback: if somehow duplicate, mark as used and continue?
            # This shouldn't happen.
            pass

    # Polish pass: try to improve by checking if we can replace some blocks
    # with better ones, or add targeted blocks for remaining uncovered
    if time.time() < deadline - 5.0 and uncovered.any():
        # For each uncovered subset, find the block with max gain among unused blocks
        # that covers it, and add it
        uncov_indices = np.where(uncovered)[0]
        for u_idx in uncov_indices:
            if time.time() > deadline - 2.0:
                break
            best_b = -1
            best_g = 0
            for b in subset_to_blocks[u_idx]:
                if b not in used_blocks:
                    g = gains[b]
                    if g > best_g:
                        best_g = g
                        best_b = b
            if best_b >= 0 and best_g > 0:
                blocks_found.append(list(all_blocks[best_b]))
                used_blocks.add(best_b)
                for u in block_to_subsets[best_b]:
                    if uncovered[u]:
                        uncovered[u] = False
                        for b in subset_to_blocks[u]:
                            gains[b] -= 1
                if not uncovered.any():
                    break

    return blocks_found
