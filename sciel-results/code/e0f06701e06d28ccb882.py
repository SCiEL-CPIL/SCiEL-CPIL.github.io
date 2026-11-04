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
    deadline = start_time + 130.0

    def check_time(budget=2.0):
        return time.time() - start_time > deadline - budget

    # Precompute all t-subsets and index them
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_block = []
        for ts in itertools.combinations(b, t):
            tsubs_in_block.append(tsub_index[ts])
        block_tsubs.append(tsubs_in_block)

    # Full universe mask
    full_mask = (1 << num_tsubs) - 1

    # Block masks
    block_masks = [0] * num_blocks
    for i in range(num_blocks):
        mask = 0
        for idx in block_tsubs[i]:
            mask |= (1 << idx)
        block_masks[i] = mask

    # Greedy set cover with vectorized gain computation
    def greedy_cover():
        uncovered = full_mask
        selected_indices = []

        # Precompute: for each t-subset, which blocks cover it
        # Actually, let's just do iterative greedy with numpy for speed

        # Use a priority-based approach: maintain uncovered count per block
        # For efficiency, we'll use numpy arrays

        uncovered_counts = np.zeros(num_blocks, dtype=np.int32)
        for i in range(num_blocks):
            uncovered_counts[i] = len(block_tsubs[i])

        uncovered_mask = uncovered
        selected_set = set()

        while uncovered_mask != 0:
            if check_time(15.0):
                break

            # Find block with max gain
            best_gain = 0
            best_block = -1

            # Vectorized: compute gain for all blocks
            # gain[i] = number of bits in (uncovered & block_masks[i])
            # This is expensive to compute fully each iteration.
            # Instead, track uncovered per t-subset and update incrementally.

            # Let's use a different approach: maintain a count array
            # where count[j] = number of selected blocks that cover t-subset j
            # A t-subset is covered if count[j] > 0
            # Gain of adding block i = number of j in block_tsubs[i] with count[j] == 0

            # This requires tracking counts, which is O(num_tsubs) to initialize
            # and O(len(block_tsubs[i])) per selection.

            # For simplicity and speed, let's do a linear scan with bit operations
            # but use numpy for the bit_count equivalent.

            # Actually, Python's int.bit_count() is fast. Let's just scan.
            for i in range(num_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = i

            if best_block == -1 or best_gain == 0:
                break

            selected_indices.append(best_block)
            uncovered_mask &= ~block_masks[best_block]

        return selected_indices

    # 1-opt reduction: remove blocks that are redundant
    def local_search_remove(indices):
        indices = list(indices)
        if not indices:
            return indices

        changed = True
        while changed and not check_time(10.0):
            changed = False
            n = len(indices)
            if n < 2:
                break
            # Compute union of all
            total_union = 0
            for idx in indices:
                total_union |= block_masks[idx]

            for i in range(n):
                if check_time(3.0):
                    break
                idx_i = indices[i]
                # Union of others
                others_union = total_union
                # Check if block_i's contribution is fully covered by others
                # We need to check if every t-subset in block_i is covered by at least one other block
                # This is: (block_masks[idx_i] & ~others_union_without_i) == 0
                # But others_union_without_i is not just total_union without idx_i's bits,
                # because idx_i might share bits with others.

                # Proper check: for each t-subset in block_i, is it covered by some other block?
                # Equivalent: block_masks[idx_i] & ~(OR of block_masks[j] for j != i) == 0

                # Compute OR of all others
                others_or = 0
                for j in range(n):
                    if j != i:
                        others_or |= block_masks[indices[j]]

                if (block_masks[idx_i] & ~others_or) == 0:
                    # Redundant
                    indices.pop(i)
                    changed = True
                    break

        return indices

    # Randomized local search: remove a random block, then repair
    def randomized_local_search(initial_indices, seed=42):
        rng = random.Random(seed)
        indices = list(initial_indices)
        if not indices:
            return indices

        current_len = len(indices)
        best_indices = list(indices)
        best_len = current_len

        max_iters = 20000
        iters = 0

        while iters < max_iters and not check_time(20.0):
            iters += 1

            if len(indices) < 2:
                break

            # Remove a random block
            remove_pos = rng.randint(0, len(indices) - 1)
            removed_idx = indices[remove_pos]
            new_indices = indices[:remove_pos] + indices[remove_pos+1:]

            # Compute what's now uncovered
            union_without = 0
            for idx in new_indices:
                union_without |= block_masks[idx]

            missing = full_mask & ~union_without

            if missing == 0:
                # Still covered
                indices = new_indices
                current_len = len(indices)
                if current_len < best_len:
                    best_len = current_len
                    best_indices = list(indices)
                continue

            # Try to repair with 1 or 2 blocks
            # Find blocks that cover the most of 'missing'
            # Score each candidate block by how much of missing it covers
            best_single = -1
            best_single_gain = 0
            for i in range(num_blocks):
                gain = (missing & block_masks[i]).bit_count()
                if gain > best_single_gain:
                    best_single_gain = gain
                    best_single = i

            if best_single != -1 and best_single_gain == missing.bit_count():
                # Single block covers all missing
                candidate = new_indices + [best_single]
                if len(candidate) < current_len:
                    indices = candidate
                    current_len = len(indices)
                    if current_len < best_len:
                        best_len = current_len
                        best_indices = list(indices)
                continue

            # Try pair: find two blocks whose union covers missing
            # This is expensive. Heuristic: take top 2 by individual gain, check if union covers
            if best_single != -1:
                # Remove best_single's coverage from missing, find next best
                missing2 = missing & ~block_masks[best_single]
                if missing2 == 0:
                    pass  # Already handled above
                else:
                    second_gain = 0
                    second_block = -1
                    for i in range(num_blocks):
                        if i == best_single:
                            continue
                        gain = (missing2 & block_masks[i]).bit_count()
                        if gain > second_gain:
                            second_gain = gain
                            second_block = i

                    if second_block != -1:
                        covered_by_pair = block_masks[best_single] | block_masks[second_block]
                        if (missing & ~covered_by_pair) == 0:
                            candidate = new_indices + [best_single, second_block]
                            if len(candidate) < current_len:
                                indices = candidate
                                current_len = len(indices)
                                if current_len < best_len:
                                    best_len = current_len
                                    best_indices = list(indices)
                            elif len(candidate) == current_len and rng.random() < 0.05:
                                indices = candidate
                                current_len = len(indices)

        return best_indices

    # Main execution
    # 1. Greedy
    initial = greedy_cover()

    if not initial:
        return []

    # 2. 1-opt reduction
    reduced = local_search_remove(initial)
    best_blocks = list(reduced)
    best_len = len(best_blocks)

    # 3. Randomized local search with multiple seeds
    for seed in range(8):
        if check_time(15.0):
            break
        result = randomized_local_search(reduced, seed=seed * 137 + 7)
        if result and len(result) < best_len:
            best_len = len(result)
            best_blocks = list(result)
            # Try more 1-opt after improvement
            best_blocks = local_search_remove(best_blocks)
            best_len = len(best_blocks)

    # 4. Final 1-opt
    best_blocks = local_search_remove(best_blocks)

    return [list(all_blocks[i]) for i in best_blocks]
