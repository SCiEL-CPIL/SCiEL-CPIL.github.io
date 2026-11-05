import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]

    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)

    # Precompute numpy arrays for fast operations
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]

    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()

        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)

        selected_blocks = []

        while True:
            if time.time() > deadline - 20.0:
                break
            if not np.any(uncovered):
                break

            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))

            if gains[best_idx] == 0:
                break

            selected_blocks.append(best_idx)

            b_subs = block_subset_arrays[best_idx]
            # Use numpy to find newly covered subsets
            newly_covered = b_subs[uncovered[b_subs]]

            if len(newly_covered) == 0:
                continue

            # Update uncovered and gains efficiently
            uncovered[newly_covered] = False
            for s in newly_covered:
                for b in subset_block_arrays[s]:
                    gains[b] -= 1

        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()

        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []

        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            for b_idx in subset_block_arrays[s_idx]:
                gains[b_idx] += 1

        selected_blocks = []
        current_uncovered = uncovered_mask.copy()

        while True:
            if time.time() > deadline - 20.0:
                break
            if not np.any(current_uncovered):
                break

            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))

            if gains[best_idx] == 0:
                break

            selected_blocks.append(best_idx)

            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[current_uncovered[b_subs]]

            if len(newly_covered) == 0:
                continue

            current_uncovered[newly_covered] = False
            for s in newly_covered:
                for b in subset_block_arrays[s]:
                    gains[b] -= 1

        return selected_blocks

    def minimize(blocks_indices, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()

        current_blocks = list(blocks_indices)

        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subset_arrays[b_idx]:
                cover_count[s_idx] += 1

        improved = True
        while improved:
            if time.time() > deadline - 20.0:
                break
            improved = False
            # Randomize order for better exploration
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)

            for i in indices:
                if i >= len(current_blocks):
                    continue
                b_idx = current_blocks[i]
                is_redundant = True
                for s_idx in block_subset_arrays[b_idx]:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s_idx in block_subset_arrays[b_idx]:
                        cover_count[s_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break  # Restart scan after removal

        return current_blocks

    def swap_improve(blocks, random_seed=None):
        """Try to replace a block with another that covers the same unique subsets."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()

        current_blocks = list(blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subset_arrays[b_idx]:
                cover_count[s_idx] += 1

        # Find blocks with unique subsets (critical blocks)
        # For each block, check if it has any subset covered only by it
        improved = True
        while improved:
            if time.time() > deadline - 20.0:
                break
            improved = False
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)

            for i in indices:
                if i >= len(current_blocks):
                    continue
                b_idx = current_blocks[i]
                # Check if this block is critical (has unique subsets)
                is_critical = False
                for s_idx in block_subset_arrays[b_idx]:
                    if cover_count[s_idx] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    # This block is redundant, remove it
                    for s_idx in block_subset_arrays[b_idx]:
                        cover_count[s_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break

        return current_blocks

    best_blocks = None
    best_len = float('inf')

    trial = 0
    while time.time() < deadline - 25.0:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial)
        min_res = minimize(greedy_res, trial)

        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res

        # Strategy 2: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 30.0:
            perturbed = list(best_blocks)
            # Remove more blocks to explore larger neighborhood
            num_remove = min(max(2, len(perturbed) // 20), len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)

                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_subset_arrays[b_idx]:
                        cover_count_temp[s_idx] += 1

                uncovered_mask = (cover_count_temp == 0).astype(bool)

                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)

                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)

                # De-duplicate
                new_set = list(dict.fromkeys(new_set))

                min_res2 = minimize(new_set, trial + 2000)

                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        # Strategy 3: Swap improvement on best
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 35.0:
            swap_res = swap_improve(best_blocks, trial + 3000)
            if len(swap_res) < best_len:
                best_len = len(swap_res)
                best_blocks = swap_res

        trial += 1

    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)

    return [list(all_blocks[i]) for i in best_blocks]
