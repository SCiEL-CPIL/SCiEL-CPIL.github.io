# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Precompute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {c: i for i, c in enumerate(all_tsubs)}

    # Precompute t-1 subsets
    all_t1subs = list(itertools.combinations(range(v), t - 1))
    n_t1subs = len(all_t1subs)
    t1sub_to_idx = {c: i for i, c in enumerate(all_t1subs)}

    # tsub_for[x, s_idx] = index of {x}+S_s in all_tsubs, or -1 if x in S_s
    tsub_for = np.full((v, n_t1subs), -1, dtype=np.int32)
    for s_idx, s in enumerate(all_t1subs):
        s_set = set(s)
        for x in range(v):
            if x not in s_set:
                full = tuple(sorted(s + (x,)))
                tsub_for[x, s_idx] = tsub_to_idx[full]

    # Precompute for each t-1 subset, the list of t-subsets that extend it
    # tsub_ext[s_idx] = list of indices of t-subsets containing S_s
    tsub_ext = []
    for s_idx, s in enumerate(all_t1subs):
        s_set = set(s)
        ext = []
        for x in range(v):
            if x not in s_set:
                full = tuple(sorted(s + (x,)))
                ext.append(tsub_to_idx[full])
        tsub_ext.append(np.array(ext, dtype=np.int32))

    def _greedy(rng, dl):
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        n_uncovered = n_tsubs

        while n_uncovered > 0:
            if time.time() > dl:
                break

            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = int(rng.choice(uncovered_indices))
            seed_tsub = all_tsubs[seed_idx]
            block = list(seed_tsub)
            block_set = set(block)

            # Get t-1 subsets of current block
            t1_idxs = [t1sub_to_idx[c] for c in itertools.combinations(block, t - 1)]

            while len(block) < k:
                if time.time() > dl:
                    break

                # Compute gain for each candidate point
                gains = np.zeros(v, dtype=np.int64)
                for x in range(v):
                    if x in block_set:
                        continue
                    # For each t-1 subset of block, get the t-subset {x}+S
                    idxs = tsub_for[x, t1_idxs]
                    valid = idxs >= 0
                    if valid.any():
                        gains[x] = uncovered[idxs[valid]].sum()

                mask = np.ones(v, dtype=bool)
                for x in block:
                    mask[x] = False
                valid_pts = mask & (gains > 0)

                if not valid_pts.any():
                    candidates = np.where(mask)[0]
                    if len(candidates) == 0:
                        break
                    best_point = int(rng.choice(candidates))
                else:
                    max_gain = np.max(gains[valid_pts])
                    candidates = np.where(valid_pts & (gains == max_gain))[0]
                    best_point = int(rng.choice(candidates))

                block.append(best_point)
                block_set.add(best_point)
                t1_idxs = [t1sub_to_idx[c] for c in itertools.combinations(block, t - 1)]

            if len(block) == k:
                block.sort()
                blocks.append(tuple(block))
                # Update uncovered
                for c in itertools.combinations(block, t):
                    idx = tsub_to_idx[tuple(c)]
                    if uncovered[idx]:
                        uncovered[idx] = False
                        n_uncovered -= 1

        return [list(b) for b in blocks]

    def _local_search(blocks, dl):
        if not blocks:
            return blocks

        blocks = [list(b) for b in blocks]
        cov_count = np.zeros(n_tsubs, dtype=np.int64)
        block_masks = []
        for b in blocks:
            mask = np.zeros(n_tsubs, dtype=bool)
            for c in itertools.combinations(b, t):
                mask[tsub_to_idx[tuple(c)]] = True
            block_masks.append(mask)
            cov_count += mask

        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > dl:
                    return blocks
                mask = block_masks[i]
                if not np.any(cov_count[mask] <= 1):
                    cov_count -= mask
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                    break

        # Try to add back useful blocks? No, just remove redundant ones
        return blocks

    def _swap_local_search(blocks, dl):
        """Try swapping elements in blocks to improve coverage."""
        if not blocks:
            return blocks

        blocks = [list(b) for b in blocks]
        cov_count = np.zeros(n_tsubs, dtype=np.int64)
        block_masks = []
        for b in blocks:
            mask = np.zeros(n_tsubs, dtype=bool)
            for c in itertools.combinations(b, t):
                mask[tsub_to_idx[tuple(c)]] = True
            block_masks.append(mask)
            cov_count += mask

        # First, remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > dl:
                    return blocks
                mask = block_masks[i]
                if not np.any(cov_count[mask] <= 1):
                    cov_count -= mask
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                    break

        # Try swapping: for each block, try replacing one element with another
        # to see if it reduces total block count or improves something
        # Actually, let's try a different approach: try to merge or replace blocks
        # For now, just do the redundancy removal
        return blocks

    best_blocks = None
    best_len = float('inf')

    # Run greedy with multiple seeds
    rng0 = random.Random(42)
    baseline = _greedy(rng0, deadline)
    if baseline:
        best_blocks = baseline
        best_len = len(baseline)

    # Try more restarts if time permits
    restart = 0
    while time.time() < deadline - 40:
        restart += 1
        rng = random.Random(restart * 7919 + 13)
        blocks = _greedy(rng, deadline)
        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    # Apply local search to the best solution
    if best_blocks and time.time() < deadline - 10:
        best_blocks = _local_search(best_blocks, deadline)

    return best_blocks if best_blocks else []

# EVOLVE-BLOCK-END
