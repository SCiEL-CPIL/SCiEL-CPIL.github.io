import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)

    # 2. Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # 3. Precompute coverage masks and tset lists for each block
    block_masks = [0] * num_k
    block_tsets = [[] for _ in range(num_k)]
    for i, block in enumerate(k_blocks):
        mask = 0
        ts_list = []
        for sub in itertools.combinations(block, t):
            idx = t_index_map[sub]
            mask |= (1 << idx)
            ts_list.append(idx)
        block_masks[i] = mask
        block_tsets[i] = ts_list

    full_mask = (1 << num_t) - 1

    # 4. Build tset_to_blocks mapping for fast candidate filtering in local search
    tset_to_blocks = [[] for _ in range(num_t)]
    for b_idx, ts_list in enumerate(block_tsets):
        for ts_idx in ts_list:
            tset_to_blocks[ts_idx].append(b_idx)

    def _greedy_pass(seed):
        """One greedy set cover pass with optional randomization."""
        if time.time() > deadline:
            return None

        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()

        while uncovered:
            if time.time() > deadline:
                return None

            # Find block with max coverage of uncovered subsets
            best_idx = -1
            best_count = -1

            # Randomize tie-breaking for diversity
            indices = list(range(num_k))
            if seed % 2 == 0:
                rng.shuffle(indices)

            for i in indices:
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_count:
                    best_count = gain
                    best_idx = i

            if best_idx == -1 or best_count == 0:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]

        return selected

    def _remove_redundant(selected_indices):
        """Remove redundant blocks from a valid covering using count-based check."""
        if not selected_indices:
            return selected_indices

        kept = list(selected_indices)
        changed = True

        while changed and time.time() < deadline:
            changed = False
            new_kept = []
            for i in range(len(kept)):
                idx = kept[i]
                # Check if any tset in this block has count 1 (only covered by this block)
                redundant = True
                for ts_idx in block_tsets[idx]:
                    count = 0
                    for j in range(len(kept)):
                        if j != i and ts_idx in block_tsets[kept[j]]:
                            count += 1
                    if count == 0:
                        redundant = False
                        break
                if redundant:
                    changed = True
                else:
                    new_kept.append(idx)

            if changed:
                kept = new_kept

        return kept

    def _local_search(selected_indices, max_steps=2000):
        """Local search to minimize number of blocks using count-based coverage."""
        if time.time() > deadline:
            return selected_indices

        rng = random.Random()
        current = list(selected_indices)
        in_sol = [False] * num_k
        for b in current:
            in_sol[b] = True

        cover_counts = [0] * num_t

        def update_counts(block_idx, delta):
            for ts_idx in block_tsets[block_idx]:
                cover_counts[ts_idx] += delta

        for b in current:
            update_counts(b, 1)

        best_sol = list(current)
        best_len = len(current)
        no_improve = 0
        max_no_improve = 50

        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break

            improved = False

            # Phase 1: Remove redundant blocks
            order = list(range(len(current)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(current):
                    continue
                b = current[pos]
                redundant = True
                for ts_idx in block_tsets[b]:
                    if cover_counts[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    update_counts(b, -1)
                    in_sol[b] = False
                    current.pop(pos)
                    improved = True
                    if len(current) < best_len:
                        best_len = len(current)
                        best_sol = list(current)
                    break

            if improved:
                no_improve = 0
                continue

            # Phase 2: 1-for-1 Swap
            if current:
                found_swap = False
                max_swap_attempts = 50
                if time.time() > deadline - 30:
                    max_swap_attempts = 20

                for _ in range(max_swap_attempts):
                    if not current:
                        break
                    pos = rng.randint(0, len(current) - 1)
                    b_out = current[pos]

                    crit_ts_list = []
                    for ts_idx in block_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            crit_ts_list.append(ts_idx)

                    if not crit_ts_list:
                        # Redundant
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        current.pop(pos)
                        improved = True
                        found_swap = True
                        break

                    # Find the critical tset with the fewest covering blocks
                    min_ts_idx = crit_ts_list[0]
                    min_len = len(tset_to_blocks[min_ts_idx])
                    for ts_idx in crit_ts_list[1:]:
                        curr_len = len(tset_to_blocks[ts_idx])
                        if curr_len < min_len:
                            min_ts_idx = ts_idx
                            min_len = curr_len

                    # Filter candidates from the shortest list
                    critical_mask = 0
                    for ts_idx in crit_ts_list:
                        critical_mask |= (1 << ts_idx)

                    valid_candidates = []
                    for b_idx in tset_to_blocks[min_ts_idx]:
                        if in_sol[b_idx]:
                            continue
                        if (block_masks[b_idx] & critical_mask) == critical_mask:
                            valid_candidates.append(b_idx)

                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        current[pos] = b_in
                        found_swap = True
                        improved = True
                        break

                if found_swap:
                    no_improve = 0
                    if len(current) < best_len:
                        best_len = len(current)
                        best_sol = list(current)
                    continue
            else:
                break

            no_improve += 1
            if no_improve > max_no_improve:
                break

        return best_sol

    def _polish(selected_indices):
        """Final polishing: repeated redundancy removal and local swaps."""
        if time.time() > deadline:
            return selected_indices

        result = _remove_redundant(selected_indices)

        for _ in range(3):
            if time.time() > deadline - 1.0:
                break
            result = _local_search(result, max_steps=1000)
            result = _remove_redundant(result)

        return result

    best_blocks = None
    best_len = float('inf')

    # Phase 1: Run multiple randomized greedy passes
    attempt = 0
    while time.time() < deadline - 20.0:
        selected = _greedy_pass(attempt)
        if selected is None:
            break

        selected = _remove_redundant(selected)

        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]

        if best_len <= 340:
            break

        attempt += 1

    # Phase 2: Polish the best solution found
    if best_blocks is not None and time.time() < deadline - 1.0:
        block_to_idx = {tuple(b): i for i, b in enumerate(k_blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_blocks]

        polished = _polish(best_indices)
        if len(polished) < best_len:
            best_len = len(polished)
            best_blocks = [list(k_blocks[i]) for i in polished]

    # Fallback if no solution found
    if best_blocks is None:
        uncovered = full_mask
        selected = []
        selected_set = set()
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = -1
            for i in range(num_k):
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_count:
                    best_count = gain
                    best_idx = i
            if best_idx == -1 or best_count == 0:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]

        if selected:
            selected = _remove_redundant(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        else:
            best_blocks = []

    return best_blocks
