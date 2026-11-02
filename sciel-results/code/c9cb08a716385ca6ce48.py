"""Covering design C(v,k,t) with vectorized greedy set cover + local search."""
import itertools
import random
import time
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 112  # safety margin before 150s kill
    start_time = time.time()

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute coverage matrix: cov_matrix[b, i] = 1 if t-sub i is in block b
    # cov_matrix is (n_blocks, n_tsubs)
    cov_matrix = np.zeros((n_blocks, n_tsubs), dtype=np.int8)
    for b_idx, block in enumerate(all_blocks):
        block_set = set(block)
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                cov_matrix[b_idx, idx] = 1

    # Also store blocks as list of tuples for output
    blocks_list = [list(b) for b in all_blocks]

    def greedy_set_cover(seed, randomize=False):
        """Vectorized greedy set cover using precomputed coverage matrix."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.int8)
        uncovered_count = n_tsubs
        chosen = []
        used_blocks = np.zeros(n_blocks, dtype=bool)

        while uncovered_count > 0:
            # Compute gain for each unused block
            gains = cov_matrix[~used_blocks].dot(uncovered)

            # Find best block
            best_local = np.argmax(gains)
            # Map back to global index
            unused_indices = np.where(~used_blocks)[0]
            best_block = unused_indices[best_local]

            if gains[best_local] == 0:
                # No block covers any remaining t-subset; should not happen if valid
                break

            # Add this block
            chosen.append(best_block)
            used_blocks[best_block] = True

            # Update uncovered
            new_covered = cov_matrix[best_block] & uncovered
            newly_covered = np.count_nonzero(new_covered)
            uncovered[new_covered] = 0
            uncovered_count -= newly_covered

        return chosen

    def local_search_remove(chosen_indices, deadline):
        """Remove redundant blocks from the chosen set."""
        if not chosen_indices:
            return chosen_indices

        # Build coverage count
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        for bi in chosen_indices:
            cov_count += cov_matrix[bi]

        chosen_list = list(chosen_indices)
        improved = True

        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(chosen_list)):
                if time.time() > deadline - 5:
                    break
                # Try removing block i
                test_cov = cov_count - cov_matrix[chosen_list[i]]
                if np.all(test_cov >= 1):
                    cov_count = test_cov
                    chosen_list.pop(i)
                    improved = True
                    break

        return chosen_list

    def local_search_swap(chosen_indices, deadline):
        """Try to swap a block with a better one (reduce total blocks by 1)."""
        if not chosen_indices:
            return chosen_indices

        chosen_set = set(chosen_indices)
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        for bi in chosen_indices:
            cov_count += cov_matrix[bi]

        chosen_list = list(chosen_indices)
        improved = True

        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(chosen_list)):
                if time.time() > deadline - 5:
                    break
                block_i = chosen_list[i]
                # Remove block_i
                test_cov = cov_count - cov_matrix[block_i]
                # Find uncovered t-subsets
                uncovered_mask = (test_cov == 0).astype(np.int8)
                if np.all(test_cov >= 1):
                    # Can just remove
                    cov_count = test_cov
                    chosen_list.pop(i)
                    improved = True
                    break

                # Find a block that covers all uncovered t-subsets and is not in chosen_set
                # A block j covers all uncovered if cov_matrix[j] & uncovered_mask == uncovered_mask
                # i.e., for all i where uncovered_mask[i]==1, cov_matrix[j,i]==1
                # This is: np.all(cov_matrix[j, uncovered_mask==1])
                uncovered_indices = np.where(uncovered_mask)[0]
                if len(uncovered_indices) == 0:
                    continue

                # Check all unused blocks
                unused_mask = np.ones(n_blocks, dtype=bool)
                for ci in chosen_list:
                    unused_mask[ci] = False

                # For each unused block, check if it covers all uncovered t-subsets
                # cov_matrix[unused, uncovered_indices] should be all 1s
                sub_matrix = cov_matrix[np.where(unused_mask)[0], uncovered_indices]
                if sub_matrix.size > 0:
                    all_covered = np.all(sub_matrix == 1, axis=1)
                    good_blocks = np.where(unused_mask)[0][all_covered]
                    if len(good_blocks) > 0:
                        # Pick the one that covers the most (for future flexibility)
                        # Actually any works; pick first
                        new_block = good_blocks[0]
                        cov_count = test_cov + cov_matrix[new_block]
                        chosen_list[i] = new_block
                        improved = True
                        break

        return chosen_list

    # Run greedy set cover with multiple seeds
    best_blocks = None
    best_count = float('inf')

    # Deterministic first
    chosen = greedy_set_cover(0, randomize=False)
    best_blocks = chosen
    best_count = len(chosen)

    # Randomized restarts (varying which block is picked on ties)
    for seed in range(1, 200):
        if time.time() > deadline - 30:
            break
        chosen = greedy_set_cover(seed, randomize=True)
        if len(chosen) < best_count:
            best_blocks = chosen
            best_count = len(chosen)

    # Local search: remove redundant blocks
    if best_blocks and time.time() < deadline - 15:
        best_blocks = local_search_remove(best_blocks, deadline)
        best_count = len(best_blocks)

    # Local search: swap to reduce
    if best_blocks and time.time() < deadline - 10:
        best_blocks = local_search_swap(best_blocks, deadline)
        best_count = len(best_blocks)

    # Second removal pass
    if best_blocks and time.time() < deadline - 5:
        best_blocks = local_search_remove(best_blocks, deadline)
        best_count = len(best_blocks)

    # Convert indices to actual blocks
    if best_blocks is None:
        best_blocks = greedy_set_cover(0, randomize=False)

    result = [blocks_list[bi] for bi in best_blocks]
    return result
