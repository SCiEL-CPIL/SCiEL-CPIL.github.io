"""Combined covering algorithm: vectorized greedy with local search refinement."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Precompute all t-subsets and index them
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute all k-blocks and their t-subset coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, list of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs.append(np.array(tsubs, dtype=np.int32))

    # Inverse: for each t-subset, list of blocks covering it
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
    # Convert to numpy arrays for faster access
    tsub_blocks_arr = [np.array(b, dtype=np.int32) for b in tsub_blocks]

    def greedy_cover(seed=0):
        """Vectorized greedy set cover with random tie-breaking."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []

        # Initial gain: how many uncovered t-subsets each block covers
        gain = np.array([np.sum(uncovered[ts]) for ts in block_tsubs], dtype=np.int32)

        while np.any(uncovered):
            if time.time() > deadline:
                break

            max_gain = gain.max()
            if max_gain == 0:
                break

            # Randomly pick among blocks with max gain for diversity
            candidates = np.where(gain == max_gain)[0]
            chosen = candidates[rng.randint(0, len(candidates) - 1)]

            blocks_used.append(all_blocks[chosen])

            # Update: mark t-subsets covered by chosen block as covered
            for ts_idx in block_tsubs[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Decrement gain for all blocks that cover this t-subset
                    for bi in tsub_blocks_arr[ts_idx]:
                        gain[bi] -= 1

            # Prevent re-selecting the same block
            gain[chosen] = -1

        return blocks_used

    def local_search(blocks, seed=0):
        """Remove redundant blocks and try to replace pairs with single blocks."""
        if not blocks:
            return blocks

        rng = random.Random(seed)

        # Build coverage count
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsub_indices = []
        for b in blocks:
            tsubs = [tsub_idx[ts] for ts in itertools.combinations(b, t)]
            block_tsub_indices.append(tsubs)
            for idx in tsubs:
                cover_count[idx] += 1

        # Pass 1: Remove redundant blocks (those whose all t-subsets are covered by others)
        kept = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not kept[i]:
                    continue
                redundant = True
                for idx in block_tsub_indices[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    kept[i] = False
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    changed = True

        reduced = [blocks[i] for i in range(len(blocks)) if kept[i]]
        block_tsub_indices = [block_tsub_indices[i] for i in range(len(blocks)) if kept[i]]

        # Pass 2: Try to replace pairs of blocks with a single block
        n_reduced = len(reduced)
        if n_reduced >= 2 and time.time() < deadline - 15:
            for trial in range(200):
                if time.time() > deadline:
                    break
                i, j = rng.sample(range(n_reduced), 2)
                # Get all t-subsets covered by blocks i and j
                tsubs_ij = set(block_tsub_indices[i] + block_tsub_indices[j])
                if len(tsubs_ij) > 35:  # C(7,4)=35, so a single k-block covers at most 35
                    continue
                # Find if any single block covers all of tsubs_ij
                # For each candidate block, check if it covers all tsubs in tsubs_ij
                found = False
                for bi in range(n_blocks):
                    if time.time() > deadline:
                        break
                    # Quick check: does block bi cover all tsubs in tsubs_ij?
                    # A block covers a t-subset if all elements of the t-subset are in the block
                    block_set = set(all_blocks[bi])
                    covers_all = True
                    for ts_idx in tsubs_ij:
                        ts = all_tsubsets[ts_idx]
                        if not all(x in block_set for x in ts):
                            covers_all = False
                            break
                    if covers_all:
                        # Replace blocks i and j with block bi
                        new_blocks = [reduced[m] for m in range(n_reduced) if m not in (i, j)]
                        new_blocks.append(all_blocks[bi])
                        new_block_tsub_indices = [block_tsub_indices[m] for m in range(n_reduced) if m not in (i, j)]
                        new_block_tsub_indices.append([tsub_idx[ts] for ts in itertools.combinations(all_blocks[bi], t)])

                        # Rebuild cover_count and check validity
                        new_cover = np.zeros(n_tsubsets, dtype=np.int32)
                        for tsubs in new_block_tsub_indices:
                            for idx in tsubs:
                                new_cover[idx] += 1
                        if np.all(new_cover >= 1):
                            reduced = new_blocks
                            block_tsub_indices = new_block_tsub_indices
                            n_reduced = len(reduced)
                            found = True
                            break
                if found:
                    # After replacement, try to remove redundant blocks again
                    cover_count = np.zeros(n_tsubsets, dtype=np.int32)
                    for tsubs in block_tsub_indices:
                        for idx in tsubs:
                            cover_count[idx] += 1
                    kept2 = [True] * len(reduced)
                    changed2 = True
                    while changed2:
                        changed2 = False
                        for ii in range(len(reduced)):
                            if not kept2[ii]:
                                continue
                            redundant2 = True
                            for idx in block_tsub_indices[ii]:
                                if cover_count[idx] == 1:
                                    redundant2 = False
                                    break
                            if redundant2:
                                kept2[ii] = False
                                for idx in block_tsub_indices[ii]:
                                    cover_count[idx] -= 1
                                changed2 = True
                    reduced = [reduced[ii] for ii in range(len(reduced)) if kept2[ii]]
                    block_tsub_indices = [block_tsub_indices[ii] for ii in range(len(reduced)) if kept2[ii]]
                    n_reduced = len(reduced)

        return reduced

    best_blocks = None
    best_len = float('inf')

    # Run greedy with multiple seeds
    num_trials = 20
    for seed in range(num_trials):
        if time.time() > deadline:
            break

        blocks = greedy_cover(seed)
        if not blocks:
            continue

        blocks = local_search(blocks, seed)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

        # Early termination if we hit a very good result
        if best_len <= 135:
            break

    if best_blocks is None:
        # Fallback: run one more greedy
        best_blocks = greedy_cover(0)
        best_blocks = local_search(best_blocks, 0)

    return list(best_blocks)
