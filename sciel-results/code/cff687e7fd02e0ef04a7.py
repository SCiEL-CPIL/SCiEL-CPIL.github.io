import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Precompute all t-subsets and their indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}

    # Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)

    # Compute masks: for each block, which t-subsets does it cover?
    block_data = []
    for block in all_k_blocks:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_data.append((block, mask))

    block_masks = np.array([m for _, m in block_data], dtype=object)
    full_mask = (1 << n_t) - 1

    def greedy_construct(seed=None, use_log=False, use_weighted=False):
        """Greedy set cover with various scoring strategies."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        # Build a list of (block_idx, mask)
        remaining = list(range(n_k))
        if seed is not None:
            rng.shuffle(remaining)

        uncovered = full_mask
        selected = []

        # For weighted greedy, maintain a weight for each t-subset
        if use_weighted:
            weights = np.ones(n_t, dtype=np.float64)
            # Randomize weights slightly for diversity
            if seed is not None:
                weights = weights * (1 + 0.1 * rng.random())

        while uncovered != 0:
            if time.time() > deadline - 30:
                break

            best_idx = -1
            best_score = -1

            # Sample a subset of blocks for speed if we have many
            if len(remaining) > 5000:
                candidates = rng.sample(remaining, min(2000, len(remaining)))
            else:
                candidates = remaining

            for i in candidates:
                mask_i = block_masks[i]
                gain = (mask_i & uncovered).bit_count()
                if gain == 0:
                    continue

                if use_log:
                    total_covered = mask_i.bit_count()
                    already_covered = total_covered - gain
                    score = gain / (1 + already_covered * 0.1)
                elif use_weighted:
                    # Sum of weights of newly covered t-subsets
                    gain_mask = mask_i & uncovered
                    # Extract indices of set bits
                    idxs = []
                    m = gain_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        idxs.append(idx)
                        m ^= lsb
                    score = sum(weights[idx] for idx in idxs)
                else:
                    score = gain

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == -1:
                break

            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]

            if use_weighted:
                # Update weights: increase weight of t-subsets that are still uncovered
                # and decrease weight of covered ones
                mask_i = block_masks[best_idx]
                for idx in range(n_t):
                    if (mask_i >> idx) & 1:
                        if (uncovered >> idx) & 1:
                            weights[idx] *= 1.5
                        else:
                            weights[idx] *= 0.5

            remaining.remove(best_idx)

        return selected

    def prune_blocks(selected_indices):
        """Remove redundant blocks iteratively."""
        if not selected_indices:
            return selected_indices

        # Build cover counts
        cover_count = [0] * n_t
        for idx in selected_indices:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                cover_count[i] += 1
                m ^= lsb

        # Verify full coverage
        for i in range(n_t):
            if cover_count[i] == 0:
                return selected_indices

        current = selected_indices[:]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            new_current = []
            for idx in current:
                mask = block_masks[idx]
                is_critical = False
                m = mask
                while m:
                    lsb = m & -m
                    i = lsb.bit_length() - 1
                    if cover_count[i] == 1:
                        is_critical = True
                        break
                    m ^= lsb

                if is_critical:
                    new_current.append(idx)
                else:
                    m = mask
                    while m:
                        lsb = m & -m
                        i = lsb.bit_length() - 1
                        cover_count[i] -= 1
                        m ^= lsb
                    improved = True

            current = new_current

        return current

    def local_search(selected_indices, max_time=30):
        """Local search: try swaps and deletions."""
        if not selected_indices:
            return selected_indices

        current = prune_blocks(selected_indices)
        current_set = set(current)
        best = current[:]
        best_len = len(best)

        # Try deletion + re-pruning
        for i in range(len(current)):
            if time.time() > deadline - 15:
                break
            trial = [x for j, x in enumerate(current) if j != i]
            cov = [0] * n_t
            for idx in trial:
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    k2 = lsb.bit_length() - 1
                    cov[k2] += 1
                    m ^= lsb
            if all(c >= 1 for c in cov):
                pruned = prune_blocks(trial)
                if len(pruned) < best_len:
                    best = pruned
                    best_len = len(best)
                    current = pruned
                    current_set = set(current)

        # Try replacement: remove one, add one
        if time.time() < deadline - 20:
            for i in range(len(current)):
                if time.time() > deadline - 15:
                    break
                removed_idx = current[i]
                trial_without = [x for j, x in enumerate(current) if j != i]
                cov = [0] * n_t
                for idx in trial_without:
                    m = block_masks[idx]
                    while m:
                        lsb = m & -m
                        k2 = lsb.bit_length() - 1
                        cov[k2] += 1
                        m ^= lsb

                uncovered_mask = 0
                for k2 in range(n_t):
                    if cov[k2] == 0:
                        uncovered_mask |= (1 << k2)

                if uncovered_mask == 0:
                    pruned = prune_blocks(trial_without)
                    if len(pruned) < best_len:
                        best = pruned
                        best_len = len(best)
                        current = pruned
                        current_set = set(current)
                    continue

                best_add = -1
                best_add_gain = -1
                for j in range(n_k):
                    if j in current_set:
                        continue
                    gain = (block_masks[j] & uncovered_mask).bit_count()
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add = j
                    if gain == uncovered_mask.bit_count():
                        break

                if best_add >= 0 and best_add_gain > 0:
                    trial = trial_without + [best_add]
                    pruned = prune_blocks(trial)
                    if len(pruned) < best_len:
                        best = pruned
                        best_len = len(best)
                        current = pruned
                        current_set = set(current)

        return best

    # Phase 1: Multiple greedy constructions with different randomizations
    best_solution = None
    best_len = float('inf')

    # Deterministic greedy
    sol = greedy_construct(seed=None, use_log=False, use_weighted=False)
    sol = prune_blocks(sol)
    sol = local_search(sol)
    best_solution = sol
    best_len = len(sol)

    # Log-gain greedy
    if time.time() < deadline - 40:
        sol = greedy_construct(seed=None, use_log=True, use_weighted=False)
        sol = prune_blocks(sol)
        sol = local_search(sol)
        if len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)

    # Weighted greedy
    if time.time() < deadline - 40:
        sol = greedy_construct(seed=None, use_log=False, use_weighted=True)
        sol = prune_blocks(sol)
        sol = local_search(sol)
        if len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)

    # Randomized greedy with different seeds
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        sol = greedy_construct(seed=seed, use_log=False, use_weighted=False)
        sol = prune_blocks(sol)
        sol = local_search(sol)
        if len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)

    # Phase 2: ILS - Iterated Local Search
    rng = random.Random(42)
    ils_iter = 0
    while time.time() < deadline - 25 and ils_iter < 50:
        ils_iter += 1

        perturbed = best_solution[:]
        if len(perturbed) > 2:
            n_remove = min(rng.randint(1, 3), len(perturbed) - 1)
            to_remove = rng.sample(range(len(perturbed)), n_remove)
            to_remove_set = set(to_remove)
            perturbed = [x for i, x in enumerate(perturbed) if i not in to_remove_set]

        cov = [0] * n_t
        for idx in perturbed:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                cov[k2] += 1
                m ^= lsb

        uncovered = full_mask
        for k2 in range(n_t):
            if cov[k2] == 0:
                uncovered |= (1 << k2)

        while uncovered != 0 and time.time() < deadline - 15:
            best_idx = -1
            best_gain = -1
            for j in range(n_k):
                if j in perturbed:
                    continue
                gain = (block_masks[j] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j
            if best_idx == -1 or best_gain == 0:
                break
            perturbed.append(best_idx)
            uncovered &= ~block_masks[best_idx]

        perturbed = prune_blocks(perturbed)
        perturbed = local_search(perturbed)

        if len(perturbed) < best_len:
            best_solution = perturbed
            best_len = len(perturbed)

    # Final validation and return
    final_cov = [0] * n_t
    for idx in best_solution:
        m = block_masks[idx]
        while m:
            lsb = m & -m
            k2 = lsb.bit_length() - 1
            final_cov[k2] += 1
            m ^= lsb

    for k2 in range(n_t):
        if final_cov[k2] == 0:
            for j in range(n_k):
                if (block_masks[j] >> k2) & 1:
                    if j not in best_solution:
                        best_solution.append(j)
                    break

    blocks = [list(block_data[i][0]) for i in best_solution]

    seen = set()
    unique_blocks = []
    for b in blocks:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            unique_blocks.append(b)

    return unique_blocks
