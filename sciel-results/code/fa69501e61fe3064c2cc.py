import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0

    def elapsed():
        return time.time() - start_time

    # Precompute all k-blocks and t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    all_tsets = list(itertools.combinations(range(v), t))
    num_blocks = len(all_blocks)
    num_tsets = len(all_tsets)

    tset_index = {ts: i for i, ts in enumerate(all_tsets)}

    # block_to_tsets: list of arrays of tset indices per block
    block_to_tsets = []
    for block in all_blocks:
        covered = np.array([tset_index[ts] for ts in itertools.combinations(block, t)], dtype=np.int32)
        block_to_tsets.append(covered)

    # tset_to_blocks: list of arrays of block indices per tset
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, tsets in enumerate(block_to_tsets):
        for ts_idx in tsets:
            tset_to_blocks[ts_idx].append(b_idx)
    for i in range(num_tsets):
        tset_to_blocks[i] = np.array(tset_to_blocks[i], dtype=np.int32)

    # tset_weights: 1 / count of blocks covering it
    tset_counts = np.zeros(num_tsets, dtype=np.int32)
    for tsets in block_to_tsets:
        for ts_idx in tsets:
            tset_counts[ts_idx] += 1
    safe_counts = np.maximum(tset_counts, 1).astype(np.float64)
    tset_weights = 1.0 / safe_counts

    def greedy_set_cover(seed):
        """Randomized weighted greedy set cover. Returns list of block indices."""
        rng = random.Random(seed)
        gains = np.zeros(num_blocks, dtype=np.float64)
        for b_idx, tsets in enumerate(block_to_tsets):
            gains[b_idx] = np.sum(tset_weights[tsets])

        # Add small random perturbation to break ties / vary selection
        gains += rng.random() * 0.001

        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []
        selected_mask = np.zeros(num_blocks, dtype=np.bool_)

        while uncovered_count > 0:
            if elapsed() > deadline - 15:
                break
            # Find best unselected block
            best_idx = -1
            best_gain = -1.0
            for i in range(num_blocks):
                if not selected_mask[i] and gains[i] > best_gain:
                    best_gain = gains[i]
                    best_idx = i
            if best_idx == -1 or best_gain <= 0:
                break
            selected.append(best_idx)
            selected_mask[best_idx] = True

            # Update: for each newly covered tset, subtract weight from all blocks covering it
            for ts_idx in block_to_tsets[best_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
                    w = tset_weights[ts_idx]
                    # Vectorized gain update
                    candidates = tset_to_blocks[ts_idx]
                    mask = ~selected_mask[candidates]
                    if np.any(mask):
                        gains[candidates[mask]] -= w

        return selected

    def local_search(initial_solution, seed):
        """Local search: remove redundant, 1-for-1 swaps, 2-for-1 swaps. Returns best found."""
        rng = random.Random(seed)
        sol = list(initial_solution)

        # Compute cover counts
        cover_counts = np.zeros(num_tsets, dtype=np.int32)
        for b in sol:
            for ts in block_to_tsets[b]:
                cover_counts[ts] += 1

        # Critical count: number of blocks covering each tset that has count==1
        # Actually, for a block b, it's removable iff all its tsets have count > 1
        # Track: for each block, how many of its tsets have count == 1
        critical_count = np.zeros(len(sol), dtype=np.int32)  # indexed by position in sol
        # Map block index -> position in sol
        block_to_pos = {}
        for pos, b in enumerate(sol):
            block_to_pos[b] = pos

        for pos, b in enumerate(sol):
            cnt = 0
            for ts in block_to_tsets[b]:
                if cover_counts[ts] == 1:
                    cnt += 1
            critical_count[pos] = cnt

        best_sol = list(sol)
        best_size = len(sol)

        no_improve = 0
        max_no_improve = 30

        while elapsed() < deadline - 3 and no_improve < max_no_improve:
            improved = False

            # Phase 1: Remove redundant blocks (in random order)
            order = list(range(len(sol)))
            rng.shuffle(order)
            for pos in reversed(order):
                if pos >= len(sol):
                    continue
                if critical_count[pos] == 0:
                    b = sol[pos]
                    # Remove
                    for ts in block_to_tsets[b]:
                        cover_counts[ts] -= 1
                        if cover_counts[ts] == 1:
                            # This tset is now critical for its sole covering block
                            # Find which block covers it
                            for other_pos in range(len(sol)):
                                if other_pos != pos:
                                    other_b = sol[other_pos]
                                    if ts in block_to_tsets[other_b]:
                                        critical_count[other_pos] += 1
                                        break
                    sol.pop(pos)
                    # Rebuild block_to_pos and critical_count is now misaligned
                    # Actually this is getting complicated. Let me simplify.
                    improved = True
                    break  # Restart the removal pass

            if improved:
                # Rebuild state after removal
                # Recompute cover counts and critical counts
                cover_counts = np.zeros(num_tsets, dtype=np.int32)
                for b in sol:
                    for ts in block_to_tsets[b]:
                        cover_counts[ts] += 1
                critical_count = np.zeros(len(sol), dtype=np.int32)
                for pos, b in enumerate(sol):
                    cnt = 0
                    for ts in block_to_tsets[b]:
                        if cover_counts[ts] == 1:
                            cnt += 1
                    critical_count[pos] = cnt
                if len(sol) < best_size:
                    best_size = len(sol)
                    best_sol = list(sol)
                no_improve = 0
                continue

            # Phase 2: 1-for-1 swap
            if sol:
                b_out_pos = rng.randint(0, len(sol) - 1)
                b_out = sol[b_out_pos]

                # Find tsets that would become uncovered
                uncovered_tsets = []
                for ts in block_to_tsets[b_out]:
                    if cover_counts[ts] == 1:
                        uncovered_tsets.append(ts)

                if uncovered_tsets:
                    # Find a block covering all uncovered_tsets (excluding b_out)
                    # Use frequency counting
                    freq = {}
                    for ts in uncovered_tsets:
                        for b in tset_to_blocks[ts]:
                            if b != b_out:
                                freq[b] = freq.get(b, 0) + 1
                    target = len(uncovered_tsets)
                    candidates = [b for b, c in freq.items() if c == target]
                    if candidates:
                        b_in = rng.choice(candidates)
                        # Perform swap
                        for ts in block_to_tsets[b_out]:
                            cover_counts[ts] -= 1
                        for ts in block_to_tsets[b_in]:
                            cover_counts[ts] += 1
                        sol[b_out_pos] = b_in
                        improved = True

            if improved:
                no_improve = 0
            else:
                no_improve += 1

        return best_sol

    # Multiple greedy restarts with different seeds
    best_solution = None
    best_size = float('inf')

    # First greedy with seed 0
    sol0 = greedy_set_cover(0)
    if sol0:
        ls_result = local_search(sol0, 0)
        if len(ls_result) < best_size:
            best_size = len(ls_result)
            best_solution = ls_result

    # More restarts if time permits
    seed = 1
    while elapsed() < deadline - 40:
        sol = greedy_set_cover(seed)
        if sol:
            ls_result = local_search(sol, seed)
            if len(ls_result) < best_size:
                best_size = len(ls_result)
                best_solution = ls_result
        seed += 1

    if best_solution is None:
        # Fallback: should not happen
        best_solution = sol0 if sol0 else []

    result = [list(all_blocks[i]) for i in best_solution]
    return result
