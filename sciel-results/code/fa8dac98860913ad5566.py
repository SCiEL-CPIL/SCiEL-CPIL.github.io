import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145

    def elapsed():
        return time.time() - start_time

    points = list(range(v))
    t_subsets = list(itertools.combinations(points, t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    k_blocks = list(itertools.combinations(points, k))
    num_k = len(k_blocks)

    # Precompute block -> t-subset indices
    block_t_indices = []
    for block in k_blocks:
        covered = []
        for sub in itertools.combinations(block, t):
            covered.append(t_index[sub])
        block_t_indices.append(covered)

    # Precompute t-subset -> block indices
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, subs in enumerate(block_t_indices):
        for s_idx in subs:
            t_to_blocks[s_idx].append(b_idx)

    # Static weights: 1 / count of blocks covering each t-subset
    t_weights = np.zeros(num_t, dtype=np.float64)
    for i in range(num_t):
        cnt = len(t_to_blocks[i])
        if cnt > 0:
            t_weights[i] = 1.0 / cnt

    # Precompute static gain for each block
    static_gain = np.zeros(num_k, dtype=np.float64)
    for i in range(num_k):
        if block_t_indices[i]:
            static_gain[i] = np.sum(t_weights[block_t_indices[i]])

    def greedy_with_noise(noise_scale, seed_val):
        """Run greedy set cover with random noise on gains."""
        rng = random.Random(seed_val)
        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)
        gain = static_gain.copy()

        if noise_scale > 0:
            noise = np.array([rng.gauss(0, noise_scale) for _ in range(num_k)])
            gain = gain + noise

        while uncovered_count > 0 and elapsed() < deadline - 15:
            current_gains = gain.copy()
            current_gains[selected_mask] = -1.0
            best_idx = int(np.argmax(current_gains))

            if gain[best_idx] <= 0:
                break

            selected.append(best_idx)
            selected_mask[best_idx] = True

            for idx in block_t_indices[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    w = t_weights[idx]
                    for b in t_to_blocks[idx]:
                        if not selected_mask[b]:
                            gain[b] -= w

        return selected

    def remove_redundant(selected):
        """Remove blocks that don't uniquely cover any t-subset."""
        if not selected:
            return selected
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b_idx in selected:
            for s_idx in block_t_indices[b_idx]:
                cover_count[s_idx] += 1

        changed = True
        while changed and elapsed() < deadline - 10:
            changed = False
            new_selected = []
            for b_idx in selected:
                needed = False
                for s_idx in block_t_indices[b_idx]:
                    if cover_count[s_idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(b_idx)
                else:
                    changed = True
                    for s_idx in block_t_indices[b_idx]:
                        cover_count[s_idx] -= 1
            if changed:
                selected = new_selected
        return selected

    def local_search(selected):
        """1-for-1 swap local search to reduce block count."""
        if not selected:
            return selected
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b_idx in selected:
            for s_idx in block_t_indices[b_idx]:
                cover_count[s_idx] += 1

        def try_drop():
            changed = True
            while changed and elapsed() < deadline - 3:
                changed = False
                new_selected = []
                for b_idx in selected:
                    needed = False
                    for s_idx in block_t_indices[b_idx]:
                        if cover_count[s_idx] == 1:
                            needed = True
                            break
                    if needed:
                        new_selected.append(b_idx)
                    else:
                        changed = True
                        for s_idx in block_t_indices[b_idx]:
                            cover_count[s_idx] -= 1
                if changed:
                    selected[:] = new_selected
                    return True
            return False

        improved = True
        while improved and elapsed() < deadline - 3:
            improved = False
            if try_drop():
                improved = True
                continue

            for i in range(len(selected)):
                if elapsed() > deadline - 3:
                    break
                b_out = selected[i]
                unique_subs = []
                for s_idx in block_t_indices[b_out]:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)

                if not unique_subs:
                    continue

                # Find candidates that cover all unique_subs
                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = set(t_to_blocks[s_idx])
                    else:
                        candidates &= set(t_to_blocks[s_idx])

                if not candidates:
                    continue
                candidates.discard(b_out)
                if not candidates:
                    continue

                # Pick best candidate
                best_in = None
                best_score = float('inf')
                best_new_unique = -1
                for b_in in candidates:
                    redundant = 0
                    new_unique = 0
                    for s_idx in block_t_indices[b_in]:
                        if cover_count[s_idx] > 1:
                            redundant += 1
                        elif cover_count[s_idx] == 0:
                            new_unique += 1
                    score = redundant
                    if score < best_score or (score == best_score and new_unique > best_new_unique):
                        best_score = score
                        best_new_unique = new_unique
                        best_in = b_in

                if best_in is None:
                    continue

                # Perform swap
                for s_idx in block_t_indices[b_out]:
                    cover_count[s_idx] -= 1
                for s_idx in block_t_indices[best_in]:
                    cover_count[s_idx] += 1
                selected[i] = best_in

                if try_drop():
                    improved = True
                    break
                else:
                    # Revert
                    for s_idx in block_t_indices[best_in]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_t_indices[b_out]:
                        cover_count[s_idx] += 1
                    selected[i] = b_out

        return selected

    def dual_guided_greedy(seed_val, max_iters=500):
        """
        Dual-guided heuristic: maintain dual variables y for uncovered t-subsets.
        At each step, select the block that maximizes sum of y over its uncovered t-subsets.
        After selection, update y for newly covered subsets.
        This is a primal-dual style approach where dual variables guide selection.
        """
        rng = random.Random(seed_val)
        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)

        # Initialize dual variables: y[i] = 1 for uncovered, 0 for covered
        y = np.ones(num_t, dtype=np.float64)

        # Current gain for each block = sum of y over its uncovered t-subsets
        gain = np.zeros(num_k, dtype=np.float64)
        for b_idx in range(num_k):
            total = 0.0
            for s_idx in block_t_indices[b_idx]:
                if uncovered[s_idx]:
                    total += y[s_idx]
            gain[b_idx] = total

        for iteration in range(max_iters):
            if uncovered_count == 0 or elapsed() > deadline - 10:
                break

            # Select block with max gain
            current_gains = gain.copy()
            current_gains[selected_mask] = -1.0
            best_idx = int(np.argmax(current_gains))

            if gain[best_idx] <= 0:
                break

            selected.append(best_idx)
            selected_mask[best_idx] = True

            # Update: for each newly covered t-subset, decrease y and update gains
            for s_idx in block_t_indices[best_idx]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    uncovered_count -= 1
                    # Decrease y for this subset (dual update)
                    old_y = y[s_idx]
                    y[s_idx] = 0.0
                    # Update gains for all blocks covering this subset
                    delta = old_y
                    for b in t_to_blocks[s_idx]:
                        if not selected_mask[b]:
                            gain[b] -= delta

        return selected

    def polish_with_restarts(selected, num_restarts=5):
        """
        Apply multiple rounds of local search with random perturbations.
        """
        if not selected:
            return selected

        best = list(selected)
        best_size = len(best)
        rng = random.Random(12345)

        for restart in range(num_restarts):
            if elapsed() > deadline - 5:
                break

            # Perturb: remove a random fraction of blocks, then re-add greedily
            cur = list(best)
            cover_count = np.zeros(num_t, dtype=np.int32)
            for b_idx in cur:
                for s_idx in block_t_indices[b_idx]:
                    cover_count[s_idx] += 1

            # Remove some random blocks
            n_remove = max(1, len(cur) // 10)
            to_remove = rng.sample(range(len(cur)), min(n_remove, len(cur)))
            for i in sorted(to_remove, reverse=True):
                b_idx = cur[i]
                for s_idx in block_t_indices[b_idx]:
                    cover_count[s_idx] -= 1
                cur.pop(i)

            # Re-add greedily to cover any newly uncovered subsets
            uncovered = np.array([cover_count[i] == 0 for i in range(num_t)])
            uncovered_count = int(uncovered.sum())

            gain = np.zeros(num_k, dtype=np.float64)
            for b_idx in range(num_k):
                total = 0.0
                for s_idx in block_t_indices[b_idx]:
                    if uncovered[s_idx]:
                        total += 1.0
                gain[b_idx] = total

            while uncovered_count > 0 and elapsed() < deadline - 5:
                current_gains = gain.copy()
                # Exclude already selected
                for b_idx in cur:
                    current_gains[b_idx] = -1.0
                best_idx = int(np.argmax(current_gains))
                if gain[best_idx] <= 0:
                    break
                cur.append(best_idx)
                for s_idx in block_t_indices[best_idx]:
                    if uncovered[s_idx]:
                        uncovered[s_idx] = False
                        uncovered_count -= 1
                        for b in t_to_blocks[s_idx]:
                            if b not in cur:
                                gain[b] -= 1.0

            # Apply local search
            cur = local_search(cur)
            if len(cur) < best_size:
                best_size = len(cur)
                best = cur

        return best

    # Multiple restarts with different strategies
    best_selected = None
    best_size = float('inf')

    # Strategy 1: Deterministic greedy
    sel = greedy_with_noise(0.0, 42)
    sel = remove_redundant(sel)
    sel = local_search(sel)
    best_selected = sel
    best_size = len(sel)

    # Strategy 2: Dual-guided greedy
    if elapsed() < deadline - 20:
        sel = dual_guided_greedy(42)
        sel = remove_redundant(sel)
        sel = local_search(sel)
        if len(sel) < best_size:
            best_size = len(sel)
            best_selected = sel

    # Strategy 3: Randomized greedy with varying noise
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    for noise in noise_levels:
        if elapsed() > deadline - 25:
            break
        for seed in seeds:
            if elapsed() > deadline - 20:
                break
            sel = greedy_with_noise(noise, seed)
            if len(sel) >= best_size:
                continue
            sel = remove_redundant(sel)
            if len(sel) >= best_size:
                continue
            sel = local_search(sel)
            if len(sel) < best_size:
                best_size = len(sel)
                best_selected = sel

    # Strategy 4: Dual-guided with different seeds
    if elapsed() < deadline - 15:
        for seed in [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]:
            if elapsed() > deadline - 10:
                break
            sel = dual_guided_greedy(seed)
            if len(sel) >= best_size:
                continue
            sel = remove_redundant(sel)
            if len(sel) >= best_size:
                continue
            sel = local_search(sel)
            if len(sel) < best_size:
                best_size = len(sel)
                best_selected = sel

    # Final polish: run local_search again on best result
    if elapsed() < deadline - 5:
        best_selected = local_search(best_selected)
        best_size = len(best_selected)

    # One more drop pass
    if elapsed() < deadline - 3:
        best_selected = remove_redundant(best_selected)
        best_size = len(best_selected)

    # Polish with restarts if time permits
    if elapsed() < deadline - 8:
        best_selected = polish_with_restarts(best_selected, num_restarts=3)
        best_size = len(best_selected)

    result = [list(k_blocks[i]) for i in best_selected]
    return result
