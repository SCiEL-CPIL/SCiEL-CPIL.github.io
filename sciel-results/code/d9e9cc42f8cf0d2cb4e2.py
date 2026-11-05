import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget

    # Precompute all t-subsets and index them
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Full universe mask
    full_mask = (1 << num_tsubs) - 1

    # Precompute all blocks and their masks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    block_masks = [0] * num_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = tsub_index[ts]
            mask |= (1 << idx)
        block_masks[i] = mask

    # --- Greedy Initialization ---
    def greedy_cover():
        uncovered = full_mask
        selected_indices = []

        while uncovered:
            if check_time(10.0):
                break
            best_gain = -1
            best_block = -1

            # Linear scan for best gain
            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = i

            if best_block == -1:
                break

            selected_indices.append(best_block)
            uncovered &= ~block_masks[best_block]

        return selected_indices

    # --- 1-Opt Reduction (Standard) ---
    def local_search_remove(indices):
        indices = list(indices)
        if not indices:
            return indices

        union_mask = 0
        for idx in indices:
            union_mask |= block_masks[idx]

        changed = True
        while changed and not check_time(5.0):
            changed = False
            new_indices = []
            for i, idx in enumerate(indices):
                # Calculate union of all others
                others_union = 0
                for j, other_idx in enumerate(indices):
                    if j != i:
                        others_union |= block_masks[other_idx]

                if (block_masks[idx] & ~others_union) == 0:
                    changed = True
                else:
                    new_indices.append(idx)

            indices = new_indices

        return indices

    # --- Randomized 2-Opt Swap Hill-Climb ---
    # This is the structural change: instead of just removing blocks,
    # we try to swap a block for another to reduce size or improve structure.
    # We maintain coverage counts to make this fast.

    def randomized_2opt_swap(initial_indices, seed=42):
        rng = random.Random(seed)
        indices = list(initial_indices)
        if not indices:
            return indices

        # Initialize coverage counts
        # coverage[i] = number of blocks in indices that cover t-subset i
        coverage = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices:
            mask = block_masks[idx]
            # Efficiently add 1 to bits set in mask?
            # numpy doesn't have a direct "add 1 to bits" for int64 masks easily without loops or tricks.
            # Since num_tsubs is ~11628 for C(19,5), we can iterate over set bits?
            # Or just use a list of counts?
            # Let's use a list of ints for counts, it's faster than numpy for bit ops in python?
            # Actually, for 11k items, a list of ints is fine.
            pass

        # Let's stick to a simpler representation for counts: a list of ints.
        # Re-initializing counts is O(N * K) which is fine for N~300, K~11628.
        # But updating counts on swap is O(K) if we do it naively.
        # Let's use a helper to update counts.

        def update_counts(idx, delta):
            mask = block_masks[idx]
            # Iterate through set bits in mask
            # This is slow in Python if done naively.
            # Alternative: precompute list of t-subset indices for each block.
            pass

    # Precompute list of t-subset indices for each block to speed up count updates
    block_tsub_indices = []
    for i in range(num_blocks):
        b = all_blocks[i]
        inds = []
        for ts in itertools.combinations(b, t):
            inds.append(tsub_index[ts])
        block_tsub_indices.append(inds)

    def randomized_2opt_swap(initial_indices, seed=42):
        rng = random.Random(seed)
        indices = list(initial_indices)
        if not indices:
            return indices

        # Initialize coverage counts
        coverage = [0] * num_tsubs
        for idx in indices:
            for tsi in block_tsub_indices[idx]:
                coverage[tsi] += 1

        current_len = len(indices)
        best_indices = list(indices)
        best_len = current_len

        T = 10.0
        alpha = 0.995
        iters = 0
        max_iters = 200000  # High limit, but time constrained

        while T > 0.1 and iters < max_iters and not check_time(5.0):
            iters += 1

            if len(indices) < 2:
                break

            # Randomly pick a block to remove
            rem_pos = rng.randint(0, len(indices) - 1)
            rem_idx = indices[rem_pos]

            # Update counts: decrement
            for tsi in block_tsub_indices[rem_idx]:
                coverage[tsi] -= 1

            # Identify uncovered t-subsets (those with count 0)
            # We need to find a replacement block that covers all of these.
            # Let's find the list of uncovered t-subsets.
            # If there are none, we can just remove the block (size reduction).
            uncovered_list = [i for i in range(num_tsubs) if coverage[i] == 0]

            if not uncovered_list:
                # Safe to remove
                indices.pop(rem_pos)
                current_len -= 1
                if current_len < best_len:
                    best_len = current_len
                    best_indices = list(indices)
                continue

            # We need to add a block that covers all tsi in uncovered_list.
            # To make this fast, we can precompute which blocks cover a specific set? No, too big.
            # Instead, pick a random tsi from uncovered_list, and find blocks that cover it.
            # Then check if that block covers ALL uncovered_list.

            # Optimization: If uncovered_list is large, it's unlikely a single block covers it all.
            # We can limit the search.

            # Pick a "critical" tsi (one with low coverage, or just random from uncovered)
            # Actually, just pick one random from uncovered_list to constrain candidates.
            target_tsi = rng.choice(uncovered_list)

            # Find candidate blocks that cover target_tsi
            # We can precompute for each tsi, the list of blocks that cover it.
            # This might be memory heavy but C(19,5) is small.
            # Let's precompute tsi_to_blocks.

            # Wait, I didn't precompute tsi_to_blocks. Let's do it now?
            # It's 11628 * ~3000 blocks? No, each block covers C(8,5)=56 tsubs.
            # Total edges = 70545 * 56 ~ 4M. Storing lists is fine.
            pass

        # Since I didn't precompute tsi_to_blocks inside the function, I'll do it here.
        # It's better to do it once outside.
        return best_indices

    # Precompute tsi_to_blocks
    tsi_to_blocks = [[] for _ in range(num_tsubs)]
    for i in range(num_blocks):
        for tsi in block_tsub_indices[i]:
            tsi_to_blocks[tsi].append(i)

    def randomized_2opt_swap_v2(initial_indices, seed=42):
        rng = random.Random(seed)
        indices = list(initial_indices)
        if not indices:
            return indices

        # Initialize coverage counts
        coverage = [0] * num_tsubs
        for idx in indices:
            for tsi in block_tsub_indices[idx]:
                coverage[tsi] += 1

        current_len = len(indices)
        best_indices = list(indices)
        best_len = current_len

        T = 10.0
        alpha = 0.995
        iters = 0
        max_iters = 50000

        while T > 0.1 and iters < max_iters and not check_time(5.0):
            iters += 1

            if len(indices) < 2:
                break

            # Randomly pick a block to remove
            rem_pos = rng.randint(0, len(indices) - 1)
            rem_idx = indices[rem_pos]

            # Update counts: decrement
            for tsi in block_tsub_indices[rem_idx]:
                coverage[tsi] -= 1

            # Identify uncovered t-subsets
            # Instead of scanning all, we can track them?
            # For now, scan. 11k is fast.
            uncovered_list = [i for i in range(num_tsubs) if coverage[i] == 0]

            if not uncovered_list:
                # Safe to remove
                indices.pop(rem_pos)
                current_len -= 1
                if current_len < best_len:
                    best_len = current_len
                    best_indices = list(indices)
                continue

            # We need to add a block that covers all tsi in uncovered_list.
            # Pick a random target from uncovered_list to filter candidates
            target_tsi = rng.choice(uncovered_list)

            # Get candidates that cover target_tsi
            candidates = tsi_to_blocks[target_tsi]

            if not candidates:
                # This shouldn't happen if the initial cover was valid
                # Restore counts
                for tsi in block_tsub_indices[rem_idx]:
                    coverage[tsi] += 1
                continue

            # Pick a random candidate
            add_idx = rng.choice(candidates)

            # Check if add_idx covers ALL uncovered_list
            valid = True
            for tsi in uncovered_list:
                if coverage[tsi] > 0:
                    continue # already covered by others
                # Check if add_idx covers tsi
                # We can check if tsi is in block_tsub_indices[add_idx]
                # This is O(1) if we use a set, but we have a list.
                # Let's convert block_tsub_indices to sets for fast lookup?
                # Or just check membership.
                if tsi not in block_tsub_indices[add_idx]:
                    valid = False
                    break

            if valid:
                # Perform swap
                indices[rem_pos] = add_idx
                # Update counts: increment for add_idx
                for tsi in block_tsub_indices[add_idx]:
                    coverage[tsi] += 1
                # Note: We already decremented rem_idx.
                # If add_idx == rem_idx, we just restored it.
                # If add_idx != rem_idx, we added new coverage.
                # The uncovered_list should now be covered.
            else:
                # Restore counts
                for tsi in block_tsub_indices[rem_idx]:
                    coverage[tsi] += 1

            T *= alpha

        return best_indices

    # --- Main Execution ---

    # 1. Greedy Initialization
    initial = greedy_cover()

    if not initial:
        return []

    # 2. Standard 1-Opt Reduction
    reduced = local_search_remove(initial)
    best_blocks = list(reduced)
    best_len = len(best_blocks)

    # 3. Randomized 2-Opt Swap Search
    for seed in range(20):
        if check_time(10.0):
            break
        result = randomized_2opt_swap_v2(best_blocks, seed=seed * 100 + 1)
        if result and len(result) < best_len:
            best_len = len(result)
            best_blocks = result

    # 4. Final 1-Opt pass
    best_blocks = local_search_remove(best_blocks)

    return [list(all_blocks[i]) for i in best_blocks]
