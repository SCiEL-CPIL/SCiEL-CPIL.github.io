import itertools
import numpy as np
import time
import random
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, compute which t-subsets it covers
    k_choose_t = len(list(itertools.combinations(range(k), t)))
    block_t_indices = np.empty((n_blocks, k_choose_t), dtype=np.int32)

    for i, block in enumerate(all_blocks):
        for j, subset in enumerate(itertools.combinations(block, t)):
            block_t_indices[i, j] = t_sub_to_idx[subset]

    # Build sparse coverage matrix: rows = blocks, cols = t-subsets
    row_indices = np.repeat(np.arange(n_blocks), k_choose_t)
    col_indices = block_t_indices.flatten()
    data = np.ones(n_blocks * k_choose_t, dtype=np.int8)
    cover_matrix = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_t))

    # Build reverse index: for each t-subset, list of blocks that cover it
    subset_to_blocks = [[] for _ in range(n_t)]
    for b_idx in range(n_blocks):
        for s_idx in block_t_indices[b_idx]:
            subset_to_blocks[s_idx].append(b_idx)

    def greedy_cover(seed=0):
        """Greedy set cover using sparse matrix for gain calculation."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        used = np.zeros(n_blocks, dtype=bool)

        while uncovered.any():
            if time.time() > deadline - 5.0:
                break

            # Calculate gains: number of uncovered subsets each block covers
            gains = (cover_matrix @ uncovered.astype(np.int8)).astype(np.int32)

            # Mask out already-used blocks
            gains[used] = -1

            max_gain = gains.max()
            if max_gain <= 0:
                break

            # Find all blocks with max gain and pick one
            candidates = np.where(gains == max_gain)[0]
            if seed == 0:
                chosen = candidates[0]
            else:
                chosen = candidates[rng.randint(0, len(candidates) - 1)]

            selected.append(chosen)
            used[chosen] = True
            uncovered[block_t_indices[chosen]] = False

        return selected

    def remove_redundant(selected):
        """Remove redundant blocks iteratively."""
        if not selected:
            return selected

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cover_count[block_t_indices[b]] += 1

        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_selected = []
            for b in selected:
                sub_idx = block_t_indices[b]
                if np.min(cover_count[sub_idx]) > 1:
                    cover_count[sub_idx] -= 1
                    changed = True
                else:
                    new_selected.append(b)
            selected = new_selected

        return selected

    def local_search(selected, time_budget):
        """Deletion + swap local search using reverse indices."""
        if not selected:
            return selected

        selected = list(selected)

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cover_count[block_t_indices[b]] += 1

        # Set of selected blocks for O(1) lookup
        selected_set = set(selected)

        improved = True
        while improved and time.time() < start_time + time_budget:
            improved = False

            # Deletion pass
            changed = True
            while changed and time.time() < start_time + time_budget:
                changed = False
                for i in range(len(selected)):
                    b = selected[i]
                    sub_idx = block_t_indices[b]
                    if np.min(cover_count[sub_idx]) > 1:
                        cover_count[sub_idx] -= 1
                        selected.pop(i)
                        selected_set.discard(b)
                        changed = True
                        improved = True
                        break

            # Swap pass
            if time.time() >= start_time + time_budget:
                break

            indices = list(range(len(selected)))
            random.shuffle(indices)
            swaps_done = 0
            max_swaps = 30

            for i in indices:
                if time.time() >= start_time + time_budget or swaps_done >= max_swaps:
                    break

                old_block = selected[i]
                sub_idx = block_t_indices[old_block]

                # Find unique subsets (covered only by this block)
                unique_mask = cover_count[sub_idx] == 1
                unique_subs = sub_idx[unique_mask]

                if len(unique_subs) == 0:
                    # Not essential: try swapping with a random block to diversify
                    for _ in range(10):
                        new_block = random.randint(0, n_blocks - 1)
                        if new_block == old_block or new_block in selected_set:
                            continue

                        # Check validity: new block must not create issues
                        # Since old block has no unique subsets, any swap is valid
                        # as long as new block doesn't duplicate
                        new_sub = block_t_indices[new_block]

                        # Perform swap
                        cover_count[sub_idx] -= 1
                        cover_count[new_sub] += 1
                        selected[i] = new_block
                        selected_set.discard(old_block)
                        selected_set.add(new_block)
                        swaps_done += 1
                        improved = True
                        break
                else:
                    # Essential: find candidates that cover all unique subsets
                    # Intersection of subset_to_blocks[s] for s in unique_subs
                    if len(unique_subs) > 5:
                        # Too many constraints, skip
                        continue

                    candidates = set(subset_to_blocks[unique_subs[0]])
                    for s in unique_subs[1:]:
                        candidates.intersection_update(subset_to_blocks[s])
                        if not candidates:
                            break

                    if not candidates:
                        continue

                    candidates.discard(old_block)
                    candidates -= selected_set

                    if not candidates:
                        continue

                    # Pick best candidate: one that maximizes coverage of other subsets
                    # that are currently covered by multiple blocks (to create redundancy)
                    best_cand = None
                    best_score = -1

                    cand_list = list(candidates)
                    # Limit candidates for speed
                    if len(cand_list) > 20:
                        random.shuffle(cand_list)
                        cand_list = cand_list[:20]

                    for cand in cand_list:
                        cand_sub = block_t_indices[cand]
                        # Score: how many of cand's subsets are already covered
                        # (this creates redundancy that may enable future deletions)
                        score = np.sum(cover_count[cand_sub] > 0)
                        if score > best_score:
                            best_score = score
                            best_cand = cand

                    if best_cand is not None:
                        old_sub = block_t_indices[old_block]
                        new_sub = block_t_indices[best_cand]
                        cover_count[old_sub] -= 1
                        cover_count[new_sub] += 1
                        selected[i] = best_cand
                        selected_set.discard(old_block)
                        selected_set.add(best_cand)
                        swaps_done += 1
                        improved = True

        return selected

    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic greedy
    selected = greedy_cover(0)
    selected = remove_redundant(selected)
    if selected:
        best_blocks = selected
        best_len = len(selected)

    # 2. Randomized greedy restarts
    n_restarts = 20
    for seed in range(1, n_restarts + 1):
        if time.time() > deadline - 20.0:
            break

        selected = greedy_cover(seed)
        selected = remove_redundant(selected)
        if selected and len(selected) < best_len:
            best_blocks = selected
            best_len = len(selected)

    # 3. Local search on best solution
    if best_blocks and time.time() < deadline - 10.0:
        remaining_time = deadline - 10.0 - time.time()
        best_blocks = local_search(best_blocks, remaining_time)
        best_blocks = remove_redundant(best_blocks)

    # 4. Final redundancy removal
    best_blocks = remove_redundant(best_blocks)

    # Convert indices back to lists of points
    result_blocks = [list(all_blocks[i]) for i in best_blocks]

    return result_blocks
