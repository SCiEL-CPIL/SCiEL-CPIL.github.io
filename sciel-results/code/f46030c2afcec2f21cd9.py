import itertools
import time
import random
import numpy as np
from math import comb

_DEADLINE_S = 120.0
_MARGIN_S = 10.0


def construct(v, k, t):
    start_time = time.time()
    rng = random.Random(42)

    def remaining():
        return max(0.0, _DEADLINE_S - _MARGIN_S - (time.time() - start_time))

    # Precompute all t-subsets and assign indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute all k-blocks
    all_kblocks = list(itertools.combinations(range(v), k))
    n_kblocks = len(all_kblocks)

    # Build sparse incidence structure using numpy
    # For each k-block, compute which t-subsets it covers
    # We'll use a list of arrays for efficiency

    # Precompute t-subsets for each k-block
    block_tsubs = []
    for block in all_kblocks:
        tsubs_in_block = [tsub_index[ts] for ts in itertools.combinations(block, t) if ts in tsub_index]
        block_tsubs.append(np.array(tsubs_in_block, dtype=np.int32))

    # Greedy set cover using bitset operations
    # uncovered_mask: boolean array of length n_tsubsets
    uncovered_mask = np.ones(n_tsubsets, dtype=bool)
    selected_blocks = []
    selected_indices = set()

    # For efficiency, we'll maintain a coverage count per t-subset
    # and a list of which blocks cover each t-subset (for redundancy checking)

    while uncovered_mask.any() and remaining() > 2.0:
        # Find the block that covers the most uncovered t-subsets
        # Use vectorized numpy operations
        best_gain = -1
        best_block_idx = -1

        # Compute gain for each block: number of uncovered t-subsets it covers
        # This can be done by checking intersection with uncovered_mask
        for bi, tsubs in enumerate(block_tsubs):
            if bi in selected_indices:
                continue
            if len(tsubs) == 0:
                continue
            # Count how many of these t-subsets are still uncovered
            gain = uncovered_mask[tsubs].sum()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = bi

        if best_block_idx < 0 or best_gain == 0:
            # No improvement possible, break
            break

        # Select this block
        selected_blocks.append(list(all_kblocks[best_block_idx]))
        selected_indices.add(best_block_idx)

        # Update uncovered mask
        uncovered_mask[block_tsubs[best_block_idx]] = False

    # Convert to list of blocks (already in the right format)
    blocks = selected_blocks

    # --- Column Pruning Phase ---
    # Remove redundant blocks: a block is redundant if all t-subsets it covers
    # are also covered by other selected blocks

    if remaining() > 1.0:
        # Build coverage structure for selected blocks
        # For each selected block, get its t-subset indices
        selected_tsubs = []
        for bi in selected_indices:
            selected_tsubs.append(block_tsubs[bi])

        # Iterate until no more removals
        improved = True
        while improved and remaining() > 1.0:
            improved = False
            new_selected = []
            for i, tsubs in enumerate(selected_tsubs):
                if len(tsubs) == 0:
                    continue
                # Check if all t-subsets in this block are covered by other blocks
                is_redundant = True
                for ts_idx in tsubs:
                    # Check if any other selected block covers this t-subset
                    covered_by_other = False
                    for j, other_tsubs in enumerate(selected_tsubs):
                        if i == j:
                            continue
                        if ts_idx in other_tsubs:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_selected.append(tsubs)
                else:
                    improved = True

            if len(new_selected) < len(selected_tsubs):
                # Rebuild blocks from remaining t-subsets
                # We need to map back to original block indices
                # This is tricky - let's use a different approach
                pass

        # Simpler pruning: iterate through blocks and check redundancy
        # by checking if removing it leaves all t-subsets covered
        pruned_blocks = []
        for i, block in enumerate(blocks):
            # Temporarily remove this block and check coverage
            temp_blocks = blocks[:i] + blocks[i+1:]
            # Check if all t-subsets are still covered
            all_covered = np.zeros(n_tsubsets, dtype=bool)
            for b in temp_blocks:
                for ts in itertools.combinations(b, t):
                    if ts in tsub_index:
                        all_covered[tsub_index[ts]] = True
            if all_covered.all():
                # This block is redundant, skip it
                improved = True
            else:
                pruned_blocks.append(block)

        blocks = pruned_blocks

    # Final validation and return
    # Ensure we have a valid covering
    final_covered = np.zeros(n_tsubsets, dtype=bool)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            if ts in tsub_index:
                final_covered[tsub_index[ts]] = True

    # If not fully covered, fall back to greedy (shouldn't happen)
    if not final_covered.all():
        # Re-run greedy from scratch with more time
        uncovered_mask = ~final_covered
        for bi, tsubs in enumerate(block_tsubs):
            if bi in selected_indices:
                continue
            if len(tsubs) == 0:
                continue
            if uncovered_mask[tsubs].any():
                gain = uncovered_mask[tsubs].sum()
                if gain > 0:
                    blocks.append(list(all_kblocks[bi]))
                    selected_indices.add(bi)
                    uncovered_mask[tsubs] = False
                    if not uncovered_mask.any():
                        break

    return blocks
