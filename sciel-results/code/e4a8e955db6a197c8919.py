import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    # Precompute all k-blocks and their t-subset coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, store the list of subset indices it covers
    block_to_subsets = [[] for _ in range(n_blocks)]
    # For each subset, store which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]

    for b_idx, block in enumerate(all_blocks):
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            block_to_subsets[b_idx].append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)

    # Convert to numpy arrays for fast operations
    block_subsets_arr = [np.array(bs, dtype=np.int32) for bs in block_to_subsets]
    subset_blocks_arr = [np.array(sb, dtype=np.int32) for sb in subset_to_blocks]

    # Greedy covering with ratio-based selection
    # gain[b] = number of currently uncovered subsets in block b
    # ratio = gain[b] / (number of already-covered subsets in block b + 1)
    # This penalizes blocks that overlap heavily with already selected blocks

    uncovered = np.ones(n_subsets, dtype=bool)
    gains = np.zeros(n_blocks, dtype=np.int32)
    covered_counts = np.zeros(n_blocks, dtype=np.int32)  # how many subsets in block are already covered

    # Initial gains: for each subset, add 1 to all blocks containing it
    for u in range(n_subsets):
        for b in subset_to_blocks[u]:
            gains[b] += 1

    blocks_found = []
    used_blocks = set()

    while True:
        if time.time() > deadline:
            break

        if not uncovered.any():
            break

        # Compute ratio for each block: gain / (covered_count + 1)
        # Only consider blocks with gain > 0
        best_b = -1
        best_ratio = -1.0

        # Vectorized computation of ratios
        # For blocks with gain > 0, compute ratio
        mask = gains > 0
        if not mask.any():
            break

        # ratio = gains / (covered_counts + 1)
        # But we only want to consider blocks not yet used
        for b in range(n_blocks):
            if mask[b] and b not in used_blocks:
                ratio = gains[b] / (covered_counts[b] + 1.0)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_b = b

        if best_b < 0 or best_ratio <= 0:
            break

        # Add this block
        blocks_found.append(list(all_blocks[best_b]))
        used_blocks.add(best_b)

        # Update uncovered and gains
        for u in block_to_subsets[best_b]:
            if uncovered[u]:
                uncovered[u] = False
                # Decrement gain for all blocks containing u
                for b in subset_to_blocks[u]:
                    gains[b] -= 1
                    # This subset was newly covered, so it's now "covered" in block b
                    # But we need to track covered_counts properly
            # For subsets that were already covered, increment covered_counts
            # Actually, we need to track how many subsets in each block are covered
            # Let's update covered_counts for all subsets in this block
        # Update covered_counts for the selected block
        # The selected block now has all its subsets "covered" (some were newly covered, some were already covered)
        # For other blocks, we need to update their covered_counts for the subsets that were newly covered

        # More careful update:
        # For each subset u in block best_b:
        #   If u was uncovered, it's now covered. For all blocks b containing u, covered_counts[b] += 1
        #   If u was already covered, nothing changes for covered_counts (it was already counted)

        # Let's redo the update properly
        # First, determine which subsets in best_b were newly covered
        newly_covered = []
        for u in block_to_subsets[best_b]:
            if not uncovered[u]:  # Was just set to False, so it was newly covered
                newly_covered.append(u)

        # For each newly covered subset, increment covered_counts for all blocks containing it
        for u in newly_covered:
            for b in subset_to_blocks[u]:
                covered_counts[b] += 1

        # Early exit if all covered
        if not uncovered.any():
            break

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
                            covered_counts[b] += 1
                if not uncovered.any():
                    break

    # Local search: try to remove redundant blocks
    # For each block, check if removing it still leaves all subsets covered
    if time.time() < deadline - 3.0:
        # Build coverage count: for each subset, how many selected blocks cover it
        selected_set = set(range(len(blocks_found)))
        # Map from block index in all_blocks to its position in blocks_found
        block_list_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        selected_block_indices = [block_list_to_idx[tuple(b)] for b in blocks_found]

        # Count how many selected blocks cover each subset
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        for b_idx in selected_block_indices:
            for u in block_to_subsets[b_idx]:
                cover_count[u] += 1

        # Try to remove blocks that don't reduce coverage below 1
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(len(selected_block_indices) - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                b_idx = selected_block_indices[i]
                # Check if this block is the only one covering any subset
                redundant = True
                for u in block_to_subsets[b_idx]:
                    if cover_count[u] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove this block
                    selected_block_indices.pop(i)
                    # Update cover_count
                    for u in block_to_subsets[b_idx]:
                        cover_count[u] -= 1
                    improved = True
                    break  # Restart the loop since indices changed

        # Rebuild blocks_found from selected_block_indices
        blocks_found = [list(all_blocks[b_idx]) for b_idx in selected_block_indices]

    # Additional local search: try to swap blocks to reduce total count
    # For each pair of blocks, see if removing one and adding another reduces coverage gaps
    if time.time() < deadline - 5.0:
        # Try to find blocks that can be swapped for better coverage
        # For each block in the current solution, try replacing it with a block that covers more uncovered subsets
        # But since all subsets should be covered, try to find blocks that allow removal of other blocks

        # Alternative: try to add a block that covers many subsets currently covered only once,
        # then remove a block that becomes redundant

        # Rebuild cover_count
        block_list_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        selected_block_indices = [block_list_to_idx[tuple(b)] for b in blocks_found]
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        for b_idx in selected_block_indices:
            for u in block_to_subsets[b_idx]:
                cover_count[u] += 1

        # For each candidate block not in the solution, compute how many subsets it covers
        # that are currently covered exactly once
        # If adding it allows removing more blocks than it adds, it's a win

        # This is complex, so let's do a simpler heuristic:
        # Try removing each block and see if we can cover the gaps with fewer new blocks

        # Actually, let's just do multiple passes of the redundant block removal
        # and also try a different greedy ordering

    # Final check: ensure all subsets are covered
    # If not, add blocks for remaining uncovered subsets
    if time.time() < deadline - 1.0:
        # Rebuild coverage
        block_list_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        selected_block_indices = [block_list_to_idx[tuple(b)] for b in blocks_found]
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        for b_idx in selected_block_indices:
            for u in block_to_subsets[b_idx]:
                cover_count[u] += 1

        # Find uncovered subsets
        uncov = np.where(cover_count == 0)[0]
        for u_idx in uncov:
            if time.time() > deadline - 0.5:
                break
            # Find any block that covers this subset
            for b in subset_to_blocks[u_idx]:
                if b not in [block_list_to_idx[tuple(bl)] for bl in blocks_found]:
                    blocks_found.append(list(all_blocks[b]))
                    break

    return blocks_found
