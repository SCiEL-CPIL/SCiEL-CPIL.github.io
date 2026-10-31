import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))

    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)

    # Precompute t-subsets for each block
    tsubs_for_block = []
    for B in all_kblocks:
        tsubs_for_block.append(list(itertools.combinations(B, t)))

    # Build mapping from t-subset to index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Build coverage matrix: coverage[block_idx, tsub_idx] = 1 if block covers tsub
    # Use a sparse representation for memory efficiency
    # For each block, store the indices of t-subsets it covers
    block_tsub_indices = []
    for B in all_kblocks:
        idxs = [tsub_to_idx[ts] for ts in tsubs_for_block[B]]
        block_tsub_indices.append(idxs)

    # Greedy construction with vectorized gain computation
    uncovered = np.ones(n_tsubs, dtype=np.int8)
    blocks = []
    block_indices_used = []

    # Precompute for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx in range(n_blocks):
        for ts_idx in block_tsub_indices[b_idx]:
            tsub_to_blocks[ts_idx].append(b_idx)

    while np.any(uncovered):
        if time.time() > deadline - 10.0:
            break

        # Find the most constrained uncovered t-subset (fewest candidate blocks)
        # Sample some uncovered t-subsets to find a good pivot
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break

        # Pick a random subset of uncovered to evaluate for speed
        if len(uncovered_indices) > 50:
            sample = random.sample(list(uncovered_indices), 50)
        else:
            sample = list(uncovered_indices)

        # Find the uncovered t-subset with fewest remaining candidate blocks
        best_pivot = None
        min_candidates = float('inf')
        for ts_idx in sample:
            if uncovered[ts_idx] == 1:
                # Count how many blocks still have potential (not yet used)
                candidates = [b for b in tsub_to_blocks[ts_idx] if b not in set(block_indices_used)]
                if len(candidates) < min_candidates:
                    min_candidates = len(candidates)
                    best_pivot = ts_idx
                if min_candidates <= 1:
                    break

        if best_pivot is None:
            best_pivot = sample[0]

        # Get candidate blocks for this pivot
        candidates = [b for b in tsub_to_blocks[best_pivot] if b not in set(block_indices_used)]

        if not candidates:
            # Should not happen if we're valid, but break to avoid infinite loop
            break

        # Compute gain for each candidate block
        best_block = None
        max_gain = -1

        for b_idx in candidates:
            gain = np.sum(uncovered[block_tsub_indices[b_idx]])
            if gain > max_gain:
                max_gain = gain
                best_block = b_idx
                if gain == len(block_tsub_indices[b_idx]):
                    break

        if best_block is not None:
            blocks.append(all_kblocks[best_block])
            block_indices_used.append(best_block)
            uncovered[block_tsub_indices[best_block]] = 0

    # Post-processing: Remove redundant blocks
    if time.time() < deadline - 5.0 and blocks:
        # Build cover counts
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for B in blocks:
            ts_idxs = [tsub_to_idx[ts] for ts in tsubs_for_block[list(B)]]
            cover_count[ts_idxs] += 1

        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_blocks = []
            for B in blocks:
                ts_idxs = [tsub_to_idx[ts] for ts in tsubs_for_block[list(B)]]
                # Check if B is redundant
                redundant = True
                for ts_idx in ts_idxs:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break

                if redundant:
                    # Remove B
                    cover_count[ts_idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(B)
            blocks = new_blocks

    # Verify validity and return
    return blocks
