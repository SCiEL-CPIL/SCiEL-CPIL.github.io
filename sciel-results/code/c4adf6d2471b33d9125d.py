"""Covering design C(v,k,t) via bitmask-accelerated greedy with repair."""
import time
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute all t-subsets as bitmasks and map to indices
    # A t-subset is covered if it's a subset of some block's bitmask
    # We track uncovered t-subsets by their index in the full list

    # Total number of t-subsets
    total_t_subsets = 1
    for i in range(t):
        total_t_subsets *= (v - i)
        total_t_subsets //= (i + 1)

    # Generate all t-subsets and their bitmasks
    t_subsets = list(itertools.combinations(range(v), t))
    t_masks = np.array([sum(1 << x for x in s) for s in t_subsets], dtype=np.uint64)
    n_t = len(t_subsets)
    valid_mask = (1 << v) - 1

    # uncovered: boolean array, True if not yet covered
    uncovered = np.ones(n_t, dtype=bool)
    n_uncovered = n_t

    blocks = []
    seen_blocks = set()

    # Precompute: for each point x, the list of t-subset indices that contain x
    # This helps compute marginal gain faster
    # point_to_subsets[x] = list of indices i where t_subsets[i] contains x
    point_to_subsets = [[] for _ in range(v)]
    for i, s in enumerate(t_subsets):
        for x in s:
            point_to_subsets[x].append(i)

    def time_check():
        return time.time() < deadline

    # Greedy phase: repeatedly pick a block that covers the most uncovered t-subsets
    # For efficiency, we build blocks greedily point-by-point

    # Strategy: 
    # 1. Pick an uncovered t-subset to seed the block
    # 2. Greedily add points that maximize newly-covered uncovered t-subsets
    # 3. Use bitmask subset check: t-subset mask is covered by block mask if (t_mask & ~block_mask) == 0

    while n_uncovered > 0 and time_check():
        # Pick the first uncovered t-subset as seed
        idx = int(np.argmax(uncovered))
        if not uncovered[idx]:
            break
        seed = t_subsets[idx]
        block = list(seed)
        block_mask = 0
        for x in seed:
            block_mask |= (1 << x)

        # Greedily add points until block has k points
        while len(block) < k and time_check():
            best_x = -1
            best_gain = 0
            block_mask_local = block_mask
            for x in range(v):
                if x in block:
                    continue
                new_mask = block_mask_local | (1 << x)
                # Count how many uncovered t-subsets are subsets of new_mask but not block_mask_local
                # A t-subset is newly covered if: uncovered[i] and (t_masks[i] & ~new_mask) == 0 and (t_masks[i] & ~block_mask_local) != 0
                
                # Vectorized:
                # Mask the complements to avoid Python negative int to uint64 overflow
                new_mask_complement = (~new_mask) & valid_mask
                old_mask_complement = (~block_mask_local) & valid_mask
                covered_by_new = (t_masks & new_mask_complement) == 0
                not_covered_by_old = (t_masks & old_mask_complement) != 0
                gain = np.count_nonzero(uncovered & covered_by_new & not_covered_by_old)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                # No point helps; pick any unused point
                for x in range(v):
                    if x not in block:
                        best_x = x
                        break
            if best_x < 0:
                break
            block.append(best_x)
            block_mask |= (1 << best_x)

        block.sort()
        block_tuple = tuple(block)
        if block_tuple in seen_blocks:
            continue
        seen_blocks.add(block_tuple)
        blocks.append(block)

        # Update uncovered: mark all t-subsets that are subsets of this block as covered
        block_mask_final = 0
        for x in block:
            block_mask_final |= (1 << x)
        # Mask the complement to ensure it's a valid positive integer for NumPy conversion
        newly_covered = (t_masks & (~block_mask_final & valid_mask)) == 0
        n_new = np.count_nonzero(uncovered & newly_covered)
        uncovered[newly_covered] = False
        n_uncovered -= n_new

    # Repair phase: ensure validity by adding blocks for any remaining uncovered t-subsets
    if n_uncovered > 0 and time_check():
        # For each uncovered t-subset, find a block that contains it, or create one
        # Simplest: for each uncovered t-subset, create a block containing it + random other points
        remaining_uncovered = np.where(uncovered)[0]
        for idx in remaining_uncovered:
            if not time_check():
                break
            ts = t_subsets[idx]
            # Check if any existing block covers it (shouldn't happen if uncovered is correct)
            covered = False
            for b in blocks:
                bm = 0
                for x in b:
                    bm |= (1 << x)
                if (ts_mask := sum(1 << x for x in ts)) and (ts_mask & ~bm) == 0:
                    covered = True
                    break
            if not covered:
                # Create a new block: ts + fill with other points up to k
                block = list(ts)
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block:
                        block.append(x)
                block.sort()
                bt = tuple(block)
                if bt not in seen_blocks:
                    seen_blocks.add(bt)
                    blocks.append(block)
                # Update uncovered
                bm = 0
                for x in block:
                    bm |= (1 << x)
                newly = (t_masks & ~bm) == 0
                uncovered[newly] = False

    # Final safety: verify and fix if needed
    # Check all t-subsets are covered
    if time_check():
        all_covered = True
        final_masks = []
        for b in blocks:
            bm = 0
            for x in b:
                bm |= (1 << x)
            final_masks.append(bm)
        if final_masks:
            final_masks_arr = np.array(final_masks, dtype=np.uint64)
            # For each t-subset, check if any block mask covers it
            # t_masks[i] & ~final_masks[j] == 0 for some j
            # Vectorized: for each i, any over j
            # This is O(n_t * n_blocks) which could be large, so do it in chunks
            for start in range(0, n_t, 10000):
                end = min(start + 10000, n_t)
                chunk = t_masks[start:end]
                # Shape: (chunk_size, n_blocks)
                covered_chunk = np.all((chunk[:, None] & ~final_masks_arr[None, :]) == 0, axis=1)
                if not np.all(covered_chunk):
                    all_covered = False
                    # Find uncovered ones and add blocks
                    unc_idx = np.where(~covered_chunk)[0] + start
                    for uidx in unc_idx:
                        ts = t_subsets[uidx]
                        block = list(ts)
                        for x in range(v):
                            if len(block) >= k:
                                break
                            if x not in block:
                                block.append(x)
                        block.sort()
                        bt = tuple(block)
                        if bt not in seen_blocks:
                            seen_blocks.add(bt)
                            blocks.append(block)
                        bm = 0
                        for x in block:
                            bm |= (1 << x)
                        final_masks.append(bm)
                    # Rebuild final_masks_arr
                    final_masks_arr = np.array(final_masks, dtype=np.uint64)
                    break
        else:
            all_covered = False

    return blocks
