import numpy as np
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148.0  # safety margin before 150s kill

    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Precompute: for each block, which t-subsets it covers
    block_to_subsets = []
    for block in all_blocks:
        subs = []
        for s in itertools.combinations(block, t):
            subs.append(subset_to_idx[s])
        block_to_subsets.append(np.array(subs, dtype=np.int32))

    # For each subset, which blocks cover it
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx, subs in enumerate(block_to_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)
    subset_to_blocks = [np.array(s, dtype=np.int32) for s in subset_to_blocks]

    # Incidence matrix: (num_blocks, num_subsets) sparse-friendly
    # Use a list of arrays for efficient per-block access
    # Also precompute block sizes (number of t-subsets per block)
    block_sizes = np.array([len(s) for s in block_to_subsets], dtype=np.int32)

    def greedy_cover(noise_scale, rng_seed, max_iter=None):
        """Randomized greedy set cover with noise for diversity."""
        rng = np.random.default_rng(rng_seed)
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = block_sizes.copy().astype(np.int32)
        remaining = num_subsets
        selected = []

        if max_iter is None:
            max_iter = 10000

        for _ in range(max_iter):
            if remaining == 0:
                break
            if time.time() > deadline - 2.0:
                break

            if gains.max() == 0:
                break

            # Add noise for diversity
            noise = rng.uniform(0, noise_scale, size=num_blocks)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))

            if gains[best_idx] == 0:
                break

            selected.append(best_idx)

            b_subs = block_to_subsets[best_idx]
            newly_covered_mask = uncovered[b_subs]
            n_new = np.count_nonzero(newly_covered_mask)

            if n_new == 0:
                continue

            newly_covered = b_subs[newly_covered_mask]
            uncovered[newly_covered] = False
            remaining -= n_new

            # Update gains: for each block, subtract how many of its subsets were just covered
            # Use matrix-vector product style
            reduction = np.zeros(num_blocks, dtype=np.int32)
            for s in newly_covered:
                for b in subset_to_blocks[s]:
                    reduction[b] += 1
            gains -= reduction

        return selected

    def eliminate_redundant(blocks, cover_count):
        """Remove blocks that don't uniquely cover any subset. Returns (new_blocks, new_cover_count)."""
        changed = True
        current = list(blocks)
        cc = cover_count.copy()

        while changed:
            changed = False
            new_current = []
            for b_idx in current:
                subs = block_to_subsets[b_idx]
                if len(subs) > 0 and np.any(cc[subs] == 1):
                    new_current.append(b_idx)
                else:
                    cc[subs] -= 1
                    changed = True
            current = new_current

        return current, cc

    def local_search(blocks, rng_seed, time_limit=3.0):
        """Local search: try to replace pairs of blocks with a single block, and remove redundant blocks."""
        rng = random.Random(rng_seed)
        current = list(blocks)

        if not current:
            return current

        # Build cover count
        cc = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current:
            subs = block_to_subsets[b_idx]
            if len(subs) > 0:
                cc[subs] += 1

        # First pass: eliminate redundant
        current, cc = eliminate_redundant(current, cc)

        # 2-for-1 replacement search
        ts = time.time()
        improved = True
        while improved and (time.time() - ts) < time_limit:
            improved = False
            n = len(current)
            if n < 2:
                break

            # Try random pairs
            for _ in range(min(n * 2, 100)):
                if time.time() > deadline - 2.0:
                    break
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i == j:
                    continue

                b1 = current[i]
                b2 = current[j]

                subs1 = block_to_subsets[b1]
                subs2 = block_to_subsets[b2]

                # Unique subsets for b1 and b2
                unique1 = subs1[cc[subs1] == 1] if len(subs1) > 0 else np.array([], dtype=np.int32)
                unique2 = subs2[cc[subs2] == 1] if len(subs2) > 0 else np.array([], dtype=np.int32)

                all_unique = np.concatenate([unique1, unique2]) if len(unique1) + len(unique2) > 0 else np.array([], dtype=np.int32)

                if len(all_unique) == 0:
                    # Both redundant
                    cc[subs1] -= 1
                    cc[subs2] -= 1
                    current.pop(max(i, j))
                    current.pop(min(i, j))
                    improved = True
                    break

                # Find a single block that covers all unique subsets
                all_unique_set = set(all_unique.tolist())
                first = all_unique[0]
                found = False
                for cand in subset_to_blocks[first]:
                    if cand == b1 or cand == b2:
                        continue
                    cand_subs = block_to_subsets[cand]
                    if all(s in all_unique_set or cc[s] > 1 for s in all_unique):
                        # cand covers all unique subsets
                        cc[subs1] -= 1
                        cc[subs2] -= 1
                        cc[cand_subs] += 1
                        current.pop(max(i, j))
                        current.pop(min(i, j))
                        current.append(cand)
                        improved = True
                        found = True
                        break

                if found:
                    break

        # Final redundancy elimination
        current, cc = eliminate_redundant(current, cc)

        return current

    def swap_search(blocks, rng_seed, time_limit=4.0):
        """Try to swap a block with another that covers its unique subsets."""
        rng = random.Random(rng_seed)
        current = list(blocks)

        if not current:
            return current

        cc = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current:
            subs = block_to_subsets[b_idx]
            if len(subs) > 0:
                cc[subs] += 1

        current, cc = eliminate_redundant(current, cc)

        ts = time.time()
        improved = True
        while improved and (time.time() - ts) < time_limit:
            improved = False
            n = len(current)
            if n == 0:
                break

            for _ in range(min(n, 50)):
                if time.time() > deadline - 2.0:
                    break
                i = rng.randint(0, n - 1)
                b_idx = current[i]
                subs = block_to_subsets[b_idx]

                if len(subs) == 0:
                    current.pop(i)
                    improved = True
                    break

                unique = subs[cc[subs] == 1]

                if len(unique) == 0:
                    cc[subs] -= 1
                    current.pop(i)
                    improved = True
                    break

                # Find replacement
                unique_set = set(unique.tolist())
                first = unique[0]
                best_cand = -1
                best_overlap = -1

                current_set = set(current)
                for cand in subset_to_blocks[first]:
                    if cand in current_set or cand == b_idx:
                        continue
                    cand_subs = block_to_subsets[cand]
                    if all(s in unique_set or cc[s] > 1 for s in unique):
                        overlap = np.count_nonzero(cc[cand_subs] > 0)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_cand = cand

                if best_cand >= 0:
                    cc[subs] -= 1
                    cc[block_to_subsets[best_cand]] += 1
                    current[i] = best_cand
                    improved = True
                    break

        current, cc = eliminate_redundant(current, cc)
        return current

    best_blocks = None
    best_len = float('inf')

    # Multiple strategies with different noise levels
    strategies = [
        (0.1, "low_noise"),
        (0.3, "med_noise"),
        (0.5, "high_noise"),
        (0.8, "very_high_noise"),
        (0.0, "deterministic"),
    ]

    trial = 0
    while time.time() < deadline - 5.0:
        # Pick a strategy
        noise_scale, strat_name = strategies[trial % len(strategies)]
        seed = trial * 12345 + 67890

        # Greedy cover
        greedy_res = greedy_cover(noise_scale, seed)

        if len(greedy_res) == 0:
            trial += 1
            continue

        # Local search refinement
        min_res = local_search(greedy_res, seed + 1, time_limit=3.0)
        min_res = swap_search(min_res, seed + 2, time_limit=2.0)
        min_res = local_search(min_res, seed + 3, time_limit=2.0)

        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = list(min_res)

        # If we have a good solution, try perturbation
        if best_blocks is not None and len(best_blocks) > 15 and time.time() < deadline - 15.0:
            perturbed = list(best_blocks)
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.Random(seed).sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)

                # Compute uncovered subsets
                cc_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    subs = block_to_subsets[b_idx]
                    if len(subs) > 0:
                        cc_temp[subs] += 1

                uncovered_mask = (cc_temp == 0)
                uncovered_indices = np.where(uncovered_mask)[0]

                if len(uncovered_indices) > 0:
                    # Repair: greedily cover uncovered
                    repair_rng = np.random.default_rng(seed + 999)
                    repair_selected = []
                    rem = len(uncovered_indices)
                    rem_gains = np.zeros(num_blocks, dtype=np.int32)
                    for b_idx in range(num_blocks):
                        subs = block_to_subsets[b_idx]
                        if len(subs) > 0:
                            rem_gains[b_idx] = np.count_nonzero(uncovered_mask[subs])

                    for _ in range(1000):
                        if rem == 0 or time.time() > deadline - 5.0:
                            break
                        if rem_gains.max() == 0:
                            break
                        noise = repair_rng.uniform(0, 0.2, size=num_blocks)
                        scores = rem_gains.astype(np.float64) + noise
                        best_idx = int(np.argmax(scores))
                        if rem_gains[best_idx] == 0:
                            break
                        repair_selected.append(best_idx)
                        b_subs = block_to_subsets[best_idx]
                        newly = b_subs[uncovered_mask[b_subs]]
                        n_new = len(newly)
                        if n_new == 0:
                            continue
                        uncovered_mask[newly] = False
                        rem -= n_new
                        for s in newly:
                            for b in subset_to_blocks[s]:
                                rem_gains[b] -= 1

                    new_set = [b for b in perturbed if b not in blocks_to_remove]
                    new_set.extend(repair_selected)

                    min_res2 = local_search(new_set, seed + 500, time_limit=2.0)
                    min_res2 = swap_search(min_res2, seed + 600, time_limit=1.5)

                    if len(min_res2) < best_len:
                        best_len = len(min_res2)
                        best_blocks = list(min_res2)

        trial += 1

    if best_blocks is None:
        greedy_res = greedy_cover(0.3, 0)
        best_blocks = local_search(greedy_res, 1, time_limit=3.0)

    # Final cleanup
    if best_blocks:
        cc_final = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in best_blocks:
            subs = block_to_subsets[b_idx]
            if len(subs) > 0:
                cc_final[subs] += 1
        best_blocks, _ = eliminate_redundant(best_blocks, cc_final)

    # Remove duplicates
    best_blocks = list(set(best_blocks))

    return [list(all_blocks[i]) for i in best_blocks]
