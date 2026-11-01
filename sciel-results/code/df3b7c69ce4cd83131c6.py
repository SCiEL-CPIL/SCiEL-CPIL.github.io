import numpy as np
import time
import itertools
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s

    # Precompute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-blocks
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)

    # For each block, precompute which t-subsets it covers
    block_tsub_indices = []
    for block in all_kblocks:
        tsub_idxs = [tsub_to_idx[tsub] for tsub in itertools.combinations(block, t)]
        block_tsub_indices.append(tsub_idxs)

    # Convert to numpy array for fast operations
    block_tsub_array = np.array(block_tsub_indices, dtype=np.int32)

    # Compute degree of each t-subset (how many blocks cover it)
    tsub_degree = np.zeros(num_tsubs, dtype=np.int32)
    for tsub_idxs in block_tsub_indices:
        tsub_degree[tsub_idxs] += 1

    # Greedy set cover with scarcity weighting
    uncovered = np.ones(num_tsubs, dtype=bool)
    chosen_block_indices = []
    used_blocks = set()

    # Precompute block membership for each t-subset (which blocks cover it)
    # This is expensive but helps with local search
    tsub_to_blocks = defaultdict(set)
    for bi, tsub_idxs in enumerate(block_tsub_indices):
        for tsub_idx in tsub_idxs:
            tsub_to_blocks[tsub_idx].add(bi)

    while True:
        if time.time() > deadline - 40:
            break
        if not uncovered.any():
            break

        # Compute scarcity-weighted gain for each unused block
        # Scarcity weight: 1/degree for rare t-subsets, 1 for common ones
        # This prioritizes covering rare configurations early
        gains = np.zeros(num_kblocks, dtype=np.float64)
        for bi in range(num_kblocks):
            if bi in used_blocks:
                continue
            # Sum of scarcity weights for uncovered t-subsets in this block
            tsub_idxs = block_tsub_array[bi]
            mask = uncovered[tsub_idxs]
            if not mask.any():
                continue
            # Weight by inverse degree (scarcity)
            weights = 1.0 / np.maximum(tsub_degree[tsub_idxs[mask]], 1)
            gains[bi] = np.sum(weights)

        best_block_idx = int(np.argmax(gains))
        if gains[best_block_idx] <= 0:
            break

        chosen_block_indices.append(best_block_idx)
        used_blocks.add(best_block_idx)
        uncovered[block_tsub_array[best_block_idx]] = False

    # Local search phase
    # Phase 1: Remove redundant blocks (1-removal)
    for pass_num in range(5):
        if time.time() > deadline - 15:
            break
        improved = False
        for idx in range(len(chosen_block_indices) - 1, -1, -1):
            if time.time() > deadline - 15:
                break
            bi = chosen_block_indices[idx]
            # Check if removing this block leaves any t-subset uncovered
            # A t-subset is uniquely covered if it's in this block and not in any other chosen block
            tsub_idxs = block_tsub_array[bi]
            safe_to_remove = True
            for tsub_idx in tsub_idxs:
                # Check if this t-subset is covered by any other chosen block
                other_blocks = tsub_to_blocks[tsub_idx]
                other_blocks.discard(bi)
                if not any(ob in used_blocks for ob in other_blocks):
                    safe_to_remove = False
                    break
            if safe_to_remove:
                chosen_block_indices.pop(idx)
                used_blocks.discard(bi)
                improved = True

        if not improved:
            break

    # Phase 2: 2-exchange (swap one block for another)
    for pass_num in range(3):
        if time.time() > deadline - 10:
            break
        improved = False
        for idx in range(len(chosen_block_indices)):
            if time.time() > deadline - 10:
                break
            bi = chosen_block_indices[idx]
            # Find a block that, when swapped in, maintains coverage and potentially improves
            # Try replacing bi with another block
            # For efficiency, only try a subset of candidate blocks
            candidates = []
            for candidate_bi in range(num_kblocks):
                if candidate_bi in used_blocks and candidate_bi != bi:
                    continue
                # Quick check: does candidate cover all t-subsets that bi uniquely covers?
                bi_tsubs = block_tsub_array[bi]
                candidate_tsubs = set(block_tsub_array[candidate_bi])
                # Check if all t-subsets in bi are covered by candidate or other chosen blocks
                can_replace = True
                for tsub_idx in bi_tsubs:
                    if tsub_idx not in candidate_tsubs:
                        # Check if covered by other chosen blocks
                        other_blocks = tsub_to_blocks[tsub_idx]
                        other_blocks.discard(bi)
                        other_blocks.discard(candidate_bi)
                        if not any(ob in used_blocks for ob in other_blocks):
                            can_replace = False
                            break
                if can_replace:
                    # Calculate gain: how many new t-subsets does candidate cover?
                    new_coverage = 0
                    for tsub_idx in candidate_tsubs:
                        if tsub_idx not in set(bi_tsubs):
                            # Check if not already covered
                            other_blocks = tsub_to_blocks[tsub_idx]
                            if not any(ob in used_blocks for ob in other_blocks):
                                new_coverage += 1
                    candidates.append((new_coverage, candidate_bi))

            if candidates:
                candidates.sort(reverse=True)
                best_new_cov, best_candidate = candidates[0]
                if best_new_cov > 0:
                    # Perform the swap
                    used_blocks.discard(bi)
                    used_blocks.add(best_candidate)
                    chosen_block_indices[idx] = best_candidate
                    improved = True

        if not improved:
            break

    # Convert to final result
    result = [list(all_kblocks[bi]) for bi in chosen_block_indices]

    # Final validation
    final_uncovered = np.zeros(num_tsubs, dtype=bool)
    for b in result:
        for tsub in itertools.combinations(b, t):
            final_uncovered[tsub_to_idx[tsub]] = True

    if not final_uncovered.all():
        # Fallback: add missing blocks greedily
        missing_tsubs = np.where(~final_uncovered)[0]
        for tsub_idx in missing_tsubs:
            # Find a block that covers this t-subset
            for bi in tsub_to_blocks[tsub_idx]:
                if bi not in used_blocks:
                    result.append(list(all_kblocks[bi]))
                    used_blocks.add(bi)
                    break

    return result
