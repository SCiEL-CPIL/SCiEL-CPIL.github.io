import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

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

    # Precompute: for each t-subset, list of blocks that cover it
    # This is the reverse index for efficient candidate finding
    tsub_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_blocks):
        for j in range(k_choose_t):
            tsub_idx = block_t_indices[i, j]
            tsub_to_blocks[tsub_idx].append(i)

    # Convert to numpy arrays for faster access
    tsub_blocks_list = []
    tsub_blocks_offsets = np.zeros(n_t + 1, dtype=np.int64)
    for i in range(n_t):
        tsub_blocks_list.extend(tsub_to_blocks[i])
        tsub_blocks_offsets[i + 1] = tsub_blocks_offsets[i] + len(tsub_to_blocks[i])
    tsub_blocks_arr = np.array(tsub_blocks_list, dtype=np.int32)

    def get_blocks_for_tsub(t_idx):
        start = tsub_blocks_offsets[t_idx]
        end = tsub_blocks_offsets[t_idx + 1]
        return tsub_blocks_arr[start:end]

    def greedy_cover(seed=0):
        """Greedy set cover with randomization."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        used = np.zeros(n_blocks, dtype=bool)

        while uncovered.any():
            if time.time() > deadline - 5.0:
                break

            # Calculate gains: number of uncovered subsets each block covers
            gains = np.zeros(n_blocks, dtype=np.int32)
            for i in range(n_blocks):
                if not used[i]:
                    gains[i] = np.sum(uncovered[block_t_indices[i]])

            max_gain = gains.max()
            if max_gain <= 0:
                break

            candidates = np.where(gains == max_gain)[0]
            if seed == 0:
                chosen = candidates[0]
            else:
                chosen = candidates[rng.randint(0, len(candidates) - 1)]

            selected.append(chosen)
            used[chosen] = True
            uncovered[block_t_indices[chosen]] = False

        return selected

    def grasp_construct(seed=0):
        """GRASP: construct solution by probabilistically selecting blocks with high potential."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        used = np.zeros(n_blocks, dtype=bool)
        alpha = 0.3  # Candidate list parameter

        while uncovered.any():
            if time.time() > deadline - 5.0:
                break

            # Calculate gains and overlap penalties
            gains = np.zeros(n_blocks, dtype=np.float64)
            for i in range(n_blocks):
                if not used[i]:
                    gain = np.sum(uncovered[block_t_indices[i]])
                    if gain > 0:
                        # Overlap penalty: how much does this block overlap with already selected
                        overlap = 0
                        for s in selected:
                            overlap += len(set(block_t_indices[i].tolist()) & set(block_t_indices[s].tolist()))
                        potential = gain / (1 + 0.1 * overlap)
                        gains[i] = potential

            max_potential = gains.max()
            if max_potential <= 0:
                break

            # Build candidate list: blocks with potential >= alpha * max
            threshold = alpha * max_potential
            candidates = np.where(gains >= threshold)[0]
            candidates = candidates[gains[candidates] > 0]

            if len(candidates) == 0:
                candidates = np.where(gains == max_potential)[0]

            # Weighted random selection from candidates
            weights = gains[candidates]
            weights = weights / weights.sum()
            chosen = candidates[rng.choice(len(candidates), p=weights)]

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

    def local_search_1swap(selected, max_rounds=10):
        """1-swap local search: try removing or replacing each block."""
        if not selected:
            return selected

        selected = list(selected)
        improved = True
        rnd = 0

        while improved and rnd < max_rounds and time.time() < deadline - 2.0:
            improved = False
            rnd += 1

            cover_count = np.zeros(n_t, dtype=np.int32)
            for b in selected:
                cover_count[block_t_indices[b]] += 1

            for i in range(len(selected)):
                if time.time() > deadline - 2.0:
                    break

                b = selected[i]
                sub_idx = block_t_indices[b]

                # Check if we can remove this block
                essential = sub_idx[cover_count[sub_idx] == 1]
                if len(essential) == 0:
                    # Can remove
                    cover_count[sub_idx] -= 1
                    selected.pop(i)
                    improved = True
                    break

                # Try to replace: find a block that covers all essential subsets
                # and is not already selected
                best_replace = -1
                best_score = -1

                # Get candidate blocks from the first essential subset
                first_essential = essential[0]
                candidates = get_blocks_for_tsub(first_essential)

                for cand in candidates:
                    if used.get(cand, False) or cand in selected:
                        continue
                    cand_sub = block_t_indices[cand]
                    cand_set = set(cand_sub.tolist())
                    if all(e in cand_set for e in essential):
                        # Score: additional coverage
                        additional = np.sum(cover_count[cand_sub] == 0)
                        if additional > best_score:
                            best_score = additional
                            best_replace = cand

                if best_replace != -1:
                    old_sub = block_t_indices[selected[i]]
                    new_sub = block_t_indices[best_replace]
                    cover_count[old_sub] -= 1
                    cover_count[new_sub] += 1
                    selected[i] = best_replace
                    improved = True
                    break

        return selected

    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic greedy
    selected = greedy_cover(0)
    selected = remove_redundant(selected)
    if selected:
        best_blocks = selected
        best_len = len(selected)

    # 2. GRASP restarts
    n_restarts = 40
    for seed in range(1, n_restarts + 1):
        if time.time() > deadline - 15.0:
            break

        selected = grasp_construct(seed)
        selected = remove_redundant(selected)
        if selected and len(selected) < best_len:
            best_blocks = selected
            best_len = len(selected)

    # 3. Local search on best solution
    if best_blocks and time.time() < deadline - 5.0:
        best_blocks = local_search_1swap(best_blocks, max_rounds=15)
        best_blocks = remove_redundant(best_blocks)

    # 4. Final redundancy removal
    best_blocks = remove_redundant(best_blocks)

    # Convert indices back to lists of points
    result_blocks = [list(all_blocks[i]) for i in best_blocks]

    return result_blocks
