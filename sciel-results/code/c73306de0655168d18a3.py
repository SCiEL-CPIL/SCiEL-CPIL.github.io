"""Optimized covering design C(v,k,t) via vectorized greedy + local search polish."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _all_combos(v, t):
    """Precompute all t-combinations as an int array of shape (n, t)."""
    if t == 0:
        return np.zeros((1, 0), dtype=np.int64)
    combos = list(itertools.combinations(range(v), t))
    if not combos:
        return np.zeros((0, t), dtype=np.int64)
    return np.array(combos, dtype=np.int64)


def _block_triples_indices(block, t, v):
    """Return indices (into the global t-combo array) of all t-subsets of block."""
    if t == 0:
        return np.array([0], dtype=np.int64)
    if t == 1:
        return np.array(sorted(block), dtype=np.int64)
    idx = []
    for comb in itertools.combinations(block, t):
        key = sum(c * (v ** (t - 1 - i)) for i, c in enumerate(comb))
        idx.append(key)
    return np.array(idx, dtype=np.int64)


def _encode_key(combo, v, t):
    return sum(c * (v ** (t - 1 - i)) for i, c in enumerate(combo))


def construct(v, k, t):
    deadline = time.time() + 140.0  # safety margin under 150s kill

    # Total number of t-subsets
    n_subsets = int(np.prod(np.array([(v - i) for i in range(t)], dtype=object)) //
                      int(np.prod(np.array([(j) for j in range(1, t + 1)], dtype=object)))) if t > 0 else 1

    # Precompute all t-combinations as a sorted int array
    if t == 0:
        # Degenerate: no triples to cover
        return []

    # Build mapping from encoded key to index
    combos_list = list(itertools.combinations(range(v), t))
    n_subsets = len(combos_list)
    combo_arr = np.array(combos_list, dtype=np.int64)  # shape (n_subsets, t)

    # Build key -> index dict
    key_to_idx = {}
    for i, comb in enumerate(combos_list):
        key = _encode_key(comb, v, t)
        key_to_idx[key] = i

    # uncovered: boolean array, True = still uncovered
    uncovered = np.ones(n_subsets, dtype=bool)

    # For each point x, precompute the list of t-subset indices that contain x
    # point_to_subsets[x] = np.array of indices
    point_to_subsets = []
    for x in range(v):
        indices = []
        for i in range(n_subsets):
            if x in combo_arr[i]:
                indices.append(i)
        point_to_subsets.append(np.array(indices, dtype=np.int64))

    # Precompute for each pair (x,y) the t-subsets containing both (for t=2 speedup, but general)
    # For t=3: pair_to_subsets[(x,y)] = subsets containing both x and y
    # This is expensive for large v; only do it if v is small enough
    use_pair_cache = (t == 2 and v <= 60) or (t == 3 and v <= 30)
    pair_to_subsets = {}
    if use_pair_cache:
        for x in range(v):
            for y in range(x + 1, v):
                idx_list = []
                for i in range(n_subsets):
                    row = combo_arr[i]
                    if x in row and y in row:
                        idx_list.append(i)
                pair_to_subsets[(x, y)] = np.array(idx_list, dtype=np.int64)

    blocks = []
    n_blocks = 0

    while True:
        if time.time() > deadline - 5.0:
            break

        n_uncov = int(np.sum(uncovered))
        if n_uncov == 0:
            break

        # Pick the uncovered t-subset with minimum index as seed
        uncov_indices = np.where(uncovered)[0]
        seed_idx = uncov_indices[0]
        block = list(combo_arr[seed_idx])

        # Greedily add points to block
        while len(block) < k:
            if time.time() > deadline - 5.0:
                break

            best_x = -1
            best_gain = -1
            block_set = set(block)

            for x in range(v):
                if x in block_set:
                    continue

                # Compute gain: number of uncovered t-subsets that would be covered
                # by adding x to block
                if t == 1:
                    gain = 1 if uncovered[x] else 0
                elif t == 2:
                    # t-subsets containing x and one point in block
                    if use_pair_cache:
                        total = 0
                        for b in block:
                            key = (min(x, b), max(x, b))
                            if key in pair_to_subsets:
                                idxs = pair_to_subsets[key]
                                total += int(np.sum(uncovered[idxs]))
                        gain = total
                    else:
                        gain = 0
                        for b in block:
                            pair = tuple(sorted((x, b)))
                            key = _encode_key(pair, v, t)
                            if key in key_to_idx:
                                idx = key_to_idx[key]
                                if uncovered[idx]:
                                    gain += 1
                elif t == 3:
                    # t-subsets containing x and two points in block
                    if use_pair_cache:
                        gain = 0
                        for b1, b2 in itertools.combinations(block, 2):
                            key = (min(x, b1, b2), max(x, b1, b2))
                            # Need triple containing x, b1, b2
                            # Find index of triple (sorted)
                            triple = tuple(sorted((x, b1, b2)))
                            k3 = _encode_key(triple, v, t)
                            if k3 in key_to_idx:
                                idx = key_to_idx[k3]
                                if uncovered[idx]:
                                    gain += 1
                    else:
                        gain = 0
                        for b1, b2 in itertools.combinations(block, 2):
                            triple = tuple(sorted((x, b1, b2)))
                            k3 = _encode_key(triple, v, t)
                            if k3 in key_to_idx:
                                idx = key_to_idx[k3]
                                if uncovered[idx]:
                                    gain += 1
                else:
                    # General case
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        combo = tuple(sorted(rest + (x,)))
                        key = _encode_key(combo, v, t)
                        if key in key_to_idx:
                            idx = key_to_idx[key]
                            if uncovered[idx]:
                                gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x == -1 or best_gain == 0:
                # No point adds coverage; pick arbitrary unused point
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
                if best_x == -1:
                    break

            block.append(best_x)

        if len(block) < k:
            break

        block.sort()

        # Mark covered t-subsets
        for comb in itertools.combinations(block, t):
            key = _encode_key(comb, v, t)
            if key in key_to_idx:
                idx = key_to_idx[key]
                uncovered[idx] = False

        blocks.append(block)
        n_blocks += 1

        if time.time() > deadline - 10.0:
            break

    # Local search polish: try swapping points in blocks to cover more
    if time.time() < deadline - 5.0 and blocks:
        # Recompute uncovered
        uncovered2 = np.ones(n_subsets, dtype=bool)
        for block in blocks:
            for comb in itertools.combinations(block, t):
                key = _encode_key(comb, v, t)
                if key in key_to_idx:
                    idx = key_to_idx[key]
                    uncovered2[idx] = False

        # Try to improve by swapping points in each block
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for bi, block in enumerate(blocks):
                if time.time() > deadline - 5.0:
                    break
                block_set = set(block)
                for i in range(k):
                    if time.time() > deadline - 5.0:
                        break
                    for x in range(v):
                        if x in block_set:
                            continue
                        # Try swapping block[i] with x
                        new_block = list(block)
                        new_block[i] = x
                        new_block.sort()
                        # Compute new uncovered count
                        # Subsets covered by old block
                        old_covered = set()
                        for comb in itertools.combinations(block, t):
                            old_covered.add(_encode_key(comb, v, t))
                        new_covered = set()
                        for comb in itertools.combinations(new_block, t):
                            new_covered.add(_encode_key(comb, v, t))
                        # Gain = subsets newly covered that were uncovered
                        gain = 0
                        for key in new_covered - old_covered:
                            if key in key_to_idx:
                                idx = key_to_idx[key]
                                if uncovered2[idx]:
                                    gain += 1
                        # Loss = subsets lost that are only covered by this block
                        # (simplified: just check if they become uncovered)
                        loss = 0
                        for key in old_covered - new_covered:
                            if key in key_to_idx:
                                idx = key_to_idx[key]
                                if uncovered2[idx]:
                                    loss += 1
                        if gain - loss > 0:
                            blocks[bi] = new_block
                            # Update uncovered
                            for comb in itertools.combinations(new_block, t):
                                key = _encode_key(comb, v, t)
                                if key in key_to_idx:
                                    idx = key_to_idx[key]
                                    uncovered2[idx] = False
                            for comb in itertools.combinations(block, t):
                                key = _encode_key(comb, v, t)
                                if key in key_to_idx:
                                    idx = key_to_idx[key]
                                    # Check if still covered by other blocks
                                    still_covered = False
                                    for other_block in blocks:
                                        if other_block is blocks[bi]:
                                            continue
                                        if all(c in other_block for c in comb):
                                            still_covered = True
                                            break
                                    if not still_covered:
                                        uncovered2[idx] = True
                            improved = True
                            break
                    if improved:
                        break

    return blocks
# EVOLVE-BLOCK-END
