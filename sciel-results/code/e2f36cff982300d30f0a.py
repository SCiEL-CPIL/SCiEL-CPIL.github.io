import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def time_left():
        return deadline - (time.time() - start_time)

    # Precompute all t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # For each block, compute which t-subsets it covers
    block_t_subsets = []
    for block in k_blocks:
        covered = [t_index[combo] for combo in itertools.combinations(block, t)]
        block_t_subsets.append(covered)

    # Build reverse mapping: for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)

    # Convert to numpy arrays
    t_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in t_to_blocks]
    t_cover_count = np.array([len(lst) for lst in t_to_blocks], dtype=np.int32)

    # Convert block_t_subsets to sets for fast membership in local search
    block_t_sets = [set(lst) for lst in block_t_subsets]

    def run_greedy(weight_mode='inverse', seed=42):
        """Run weighted greedy set cover."""
        if weight_mode == 'inverse':
            safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
            t_weights = 1.0 / safe_counts
        elif weight_mode == 'random':
            rng = np.random.default_rng(seed)
            t_weights = rng.random(num_t).astype(np.float64)
            safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
            t_weights = 0.5 / safe_counts + 0.5 * t_weights
        else:
            safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
            t_weights = 1.0 / safe_counts

        # Initialize gain for each block
        gain = np.zeros(num_k, dtype=np.float64)
        for i, covered in enumerate(block_t_subsets):
            if covered:
                gain[i] = np.sum(t_weights[covered])

        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)

        while uncovered_count > 0 and time_left() > 25:
            masked_gain = gain.copy()
            masked_gain[selected_mask] = -1.0
            best_idx = int(np.argmax(masked_gain))
            best_gain = gain[best_idx]

            if best_gain <= 0:
                break

            selected.append(best_idx)
            selected_mask[best_idx] = True

            for idx in block_t_subsets[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    w = t_weights[idx]
                    for b in t_to_blocks[idx]:
                        if not selected_mask[b]:
                            gain[b] -= w

        return selected

    def local_search(selected):
        """Aggressive local search: drop, then fast swaps using set operations."""
        if not selected:
            return selected

        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1

        selected_set = set(selected)
        n_sel = len(selected)

        # Phase 1: Repeated single-drop removal
        changed = True
        max_rounds_drop = 30
        for _ in range(max_rounds_drop):
            if time_left() < 12:
                break
            if not changed:
                break
            changed = False
            for si in range(len(selected) - 1, -1, -1):
                if time_left() < 12:
                    break
                block_idx = selected[si]
                can_remove = True
                for idx in block_t_subsets[block_idx]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for idx in block_t_subsets[block_idx]:
                        cover_count[idx] -= 1
                    selected.pop(si)
                    selected_set.discard(block_idx)
                    changed = True

        # Phase 2: Fast 1-for-1 swaps using set operations
        # Key optimization: for a drop candidate, find critical t-subsets,
        # then find replacement blocks that cover ALL critical subsets using set intersection
        max_swap_rounds = 50
        for round_num in range(max_swap_rounds):
            if time_left() < 10:
                break
            improved = False

            for si in range(len(selected)):
                if time_left() < 10:
                    break
                if improved:
                    break

                drop_block = selected[si]
                drop_set = block_t_subsets[drop_block]

                # Find critical t-subsets (only covered by this block)
                critical = [idx for idx in drop_set if cover_count[idx] == 1]

                if not critical:
                    # Can just remove
                    for idx in drop_set:
                        cover_count[idx] -= 1
                    selected.pop(si)
                    selected_set.discard(drop_block)
                    improved = True
                    continue

                # Build set of critical indices for fast lookup
                critical_set = set(critical)

                # Find best replacement: must cover all critical subsets
                best_candidate = -1
                best_score = -1

                # For each block, check if it covers all critical subsets
                # Using set intersection for speed
                for b in range(num_k):
                    if b in selected_set:
                        continue
                    b_set = block_t_sets[b]
                    # Check if b_set covers all critical
                    if not critical_set.issubset(b_set):
                        continue
                    # Score: number of t-subsets in b_set with cover_count == 0
                    new_unique = 0
                    for idx in b_set:
                        if cover_count[idx] == 0:
                            new_unique += 1
                    if new_unique > best_score:
                        best_score = new_unique
                        best_candidate = b
                        if best_score >= len(critical):  # Good enough, early exit
                            break

                if best_candidate != -1:
                    for idx in drop_set:
                        cover_count[idx] -= 1
                    for idx in block_t_subsets[best_candidate]:
                        cover_count[idx] += 1
                    selected[si] = best_candidate
                    selected_set.discard(drop_block)
                    selected_set.add(best_candidate)
                    improved = True

        return selected

    # Run multiple greedy restarts with different weight modes
    best_selected = None
    best_len = float('inf')

    # Restart 1: standard inverse-count weights
    sel1 = run_greedy('inverse', 42)
    sel1 = local_search(sel1)
    if len(sel1) < best_len:
        best_len = len(sel1)
        best_selected = sel1[:]

    if time_left() > 35:
        # Restart 2: perturbed weights
        sel2 = run_greedy('random', 123)
        sel2 = local_search(sel2)
        if len(sel2) < best_len:
            best_len = len(sel2)
            best_selected = sel2[:]

    if time_left() > 35:
        # Restart 3: different perturbation
        sel3 = run_greedy('random', 777)
        sel3 = local_search(sel3)
        if len(sel3) < best_len:
            best_len = len(sel3)
            best_selected = sel3[:]

    if time_left() > 35:
        # Restart 4: another perturbation
        sel4 = run_greedy('random', 999)
        sel4 = local_search(sel4)
        if len(sel4) < best_len:
            best_len = len(sel4)
            best_selected = sel4[:]

    # Final local search on best
    if best_selected and time_left() > 10:
        best_selected = local_search(best_selected)

    result = [list(k_blocks[i]) for i in best_selected]

    # Ensure validity
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in best_selected:
            for idx in block_t_subsets[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            selected_set = set(best_selected)
            for i in range(num_k):
                if time_left() < 3:
                    break
                if i in selected_set:
                    continue
                covers_uncovered = False
                for idx in block_t_subsets[i]:
                    if final_cover[idx] == 0:
                        covers_uncovered = True
                        break
                if covers_uncovered:
                    best_selected.append(i)
                    selected_set.add(i)
                    for idx2 in block_t_subsets[i]:
                        final_cover[idx2] += 1
            result = [list(k_blocks[i]) for i in best_selected]

    return result
