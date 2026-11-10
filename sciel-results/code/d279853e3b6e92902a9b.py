import itertools
import time
import random
import numpy as np
from math import comb

_DEADLINE_S = 120.0
_MARGIN_S = 15.0


def construct(v, k, t):
    start_time = time.time()

    def remaining():
        return max(0.0, _DEADLINE_S - _MARGIN_S - (time.time() - start_time))

    # Precompute all t-subsets and assign indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute all k-blocks
    all_kblocks = list(itertools.combinations(range(v), k))
    n_kblocks = len(all_kblocks)

    # Build bitset representation: for each k-block, a bitset of which t-subsets it covers
    # Use Python integers as arbitrary-precision bitsets
    # Bit i set means t-subset i is covered by this block
    block_bitsets = []
    for block in all_kblocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            mask |= (1 << idx)
        block_bitsets.append(mask)

    # Also store the t-subset indices per block for pruning
    block_tsubs = []
    for block in all_kblocks:
        tsubs_in_block = [tsub_index[ts] for ts in itertools.combinations(block, t)]
        block_tsubs.append(tsubs_in_block)

    # Greedy set cover using bitset operations
    # uncovered: bitset of uncovered t-subsets
    uncovered = (1 << n_tsubsets) - 1
    selected_indices = []

    while uncovered != 0 and remaining() > 3.0:
        # Find the block that covers the most uncovered t-subsets
        # For each block, gain = popcount(uncovered & block_bitset)
        best_gain = -1
        best_block_idx = -1

        for bi in range(n_kblocks):
            if bi in selected_indices:
                continue
            gain = bin(uncovered & block_bitsets[bi]).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block_idx = bi

        if best_block_idx < 0 or best_gain == 0:
            break

        selected_indices.append(best_block_idx)
        uncovered &= ~block_bitsets[best_block_idx]

    # Build initial block list
    blocks = [list(all_kblocks[bi]) for bi in selected_indices]

    # --- Column Pruning Phase ---
    # Remove redundant blocks using coverage counts
    if remaining() > 2.0:
        # Build coverage count: for each t-subset, how many selected blocks cover it
        coverage_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in selected_indices:
            for ts_idx in block_tsubs[bi]:
                coverage_count[ts_idx] += 1

        # Iteratively remove redundant blocks
        changed = True
        while changed and remaining() > 2.0:
            changed = False
            new_indices = []
            for bi in selected_indices:
                tsubs = block_tsubs[bi]
                # Check if all t-subsets in this block have coverage > 1
                redundant = all(coverage_count[ts_idx] > 1 for ts_idx in tsubs)
                if redundant:
                    # Remove this block
                    for ts_idx in tsubs:
                        coverage_count[ts_idx] -= 1
                    changed = True
                else:
                    new_indices.append(bi)
            selected_indices = new_indices

        blocks = [list(all_kblocks[bi]) for bi in selected_indices]

    # Final validation
    final_covered = np.zeros(n_tsubsets, dtype=bool)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            if ts in tsub_index:
                final_covered[tsub_index[ts]] = True

    if not final_covered.all():
        # Fallback: add blocks to cover remaining
        for bi in range(n_kblocks):
            if bi not in selected_indices:
                tsubs = block_tsubs[bi]
                if any(not final_covered[ts_idx] for ts_idx in tsubs):
                    blocks.append(list(all_kblocks[bi]))
                    selected_indices.append(bi)
                    for ts_idx in tsubs:
                        final_covered[ts_idx] = True
                    if final_covered.all():
                        break

    return blocks
