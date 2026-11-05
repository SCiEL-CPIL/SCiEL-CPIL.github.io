import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, the array of t-subset indices it covers
    block_t_arrays = []
    for k_sub in all_k_subs:
        arr = np.fromiter((t_sub_to_idx[ts] for ts in itertools.combinations(k_sub, t)),
                          dtype=np.int32, count=(k * (k - 1) * (k - 2) * (k - 3) * (k - 4)) // 120)
        block_t_arrays.append(arr)

    # Reverse mapping as arrays: for each t-subset, which k-subsets cover it
    t_sub_to_k_arrays = [np.array([], dtype=np.int32) for _ in range(n_t)]
    for i, arr in enumerate(block_t_arrays):
        for t_idx in arr:
            t_sub_to_k_arrays[t_idx] = np.append(t_sub_to_k_arrays[t_idx], i)
    t_sub_to_k_arrays = [np.asarray(x, dtype=np.int32) for x in t_sub_to_k_arrays]

    def greedy_cover(seed, strategy="gain"):
        """Vectorized greedy set cover."""
        rng = np.random.default_rng(seed)
        uncovered = np.ones(n_t, dtype=bool)
        gains = np.fromiter((len(a) for a in block_t_arrays), dtype=np.int32, count=n_k)

        # cover count per t-subset (for "hardest")
        t_cover_count = np.zeros(n_t, dtype=np.int32)
        if strategy == "hardest":
            for i in range(n_k):
                t_cover_count[block_t_arrays[i]] += 1

        selected = []
        for _ in range(n_k):
            if not uncovered.any():
                break

            if strategy == "gain":
                noise = rng.uniform(0.0, 0.3, size=n_k)
                scores = gains + noise
                best_idx = int(np.argmax(scores))
                if gains[best_idx] <= 0:
                    break
            else:
                unc_idx = np.where(uncovered)[0]
                if len(unc_idx) == 0:
                    break
                min_cov = int(np.min(t_cover_count[unc_idx]))
                hard_ts = unc_idx[t_cover_count[unc_idx] == min_cov]
                target_t = int(rng.choice(hard_ts))
                cands = t_sub_to_k_arrays[target_t]
                cands = cands[gains[cands] > 0]
                if len(cands) == 0:
                    break
                best_idx = int(rng.choice(cands))

            selected.append(best_idx)
            b_arr = block_t_arrays[best_idx]
            newly = b_arr[uncovered[b_arr]]
            uncovered[newly] = False
            for s in newly:
                gains[t_sub_to_k_arrays[s]] -= 1
                t_cover_count[s] -= 1

        return selected

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass."""
        if not blocks_idx:
            return blocks_idx
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            cover_count[block_t_arrays[b_idx]] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                arr = block_t_arrays[b_idx]
                is_essential = bool(np.any(cover_count[arr] == 1))
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    cover_count[arr] -= 1
            blocks_idx = new_blocks
        return blocks_idx

    def simulated_annealing(initial_blocks, max_time):
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_t_arrays[b_idx]] += 1

        num_uncovered = int(np.sum(cover_count == 0))
        penalty = 500.0
        current_energy = len(current_blocks) + penalty * num_uncovered

        best_blocks = list(current_blocks)
        best_energy = current_energy

        rng = random.Random(42)
        T = 30.0
        cooling_rate = 0.995
        min_T = 0.05

        while time.time() < max_time and T > min_T:
            move_type = rng.random()

            if move_type < 0.40 and len(current_blocks) > 0:
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                arr = block_t_arrays[b_idx]
                new_uncovered = int(np.sum(cover_count[arr] == 1))
                if new_uncovered == 0:
                    cover_count[arr] -= 1
                    current_blocks.pop(i)
                    current_energy -= 1.0
                else:
                    delta_E = 1.0 + penalty * new_uncovered
                    if rng.random() < np.exp(-delta_E / T):
                        cover_count[arr] -= 1
                        current_blocks.pop(i)
                        current_energy += delta_E

            elif move_type < 0.85 and len(current_blocks) > 0:
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                old_arr = block_t_arrays[old_b]
                unique_ts = old_arr[cover_count[old_arr] == 1]
                if len(unique_ts) > 0:
                    cands = set(t_sub_to_k_arrays[unique_ts[0]].tolist())
                    for t_idx in unique_ts[1:]:
                        cands.intersection_update(set(t_sub_to_k_arrays[t_idx].tolist()))
                    cands.discard(old_b)
                    cands -= set(current_blocks)
                else:
                    cands = set()
                    for _ in range(10):
                        c = rng.randint(0, n_k - 1)
                        if c not in current_blocks:
                            cands.add(c)
                if cands:
                    new_b = rng.choice(list(cands))
                    temp_cover = cover_count.copy()
                    temp_cover[old_arr] -= 1
                    newly_uncovered = int(np.sum(temp_cover == 0))
                    temp_cover[block_t_arrays[new_b]] += 1
                    after_add_uncovered = int(np.sum(temp_cover == 0))
                    delta_E = penalty * (after_add_uncovered - num_uncovered)
                    if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
                        cover_count[old_arr] -= 1
                        cover_count[block_t_arrays[new_b]] += 1
                        current_blocks[i] = new_b
                        current_energy += delta_E

            else:
                if num_uncovered > 0:
                    unc_idx = np.where(cover_count == 0)[0]
                    if len(unc_idx) > 0:
                        target_t = int(rng.choice(unc_idx))
                        cands = [b for b in t_sub_to_k_arrays[target_t].tolist()
                                 if b not in current_blocks]
                        if cands:
                            new_b = rng.choice(cands)
                            new_arr = block_t_arrays[new_b]
                            newly_covered = int(np.sum(cover_count[new_arr] == 0))
                            delta_E = 1.0 - penalty * newly_covered
                            if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
                                cover_count[new_arr] += 1
                                current_blocks.append(new_b)
                                current_energy += delta_E

            num_uncovered = int(np.sum(cover_count == 0))

            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)

            T *= cooling_rate

        return best_blocks

    def local_search_swap(initial_blocks, max_time):
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_t_arrays[b_idx]] += 1

        if np.any(cover_count == 0):
            return current_blocks

        current_blocks_set = set(current_blocks)
        best_blocks = list(current_blocks)
        best_len = len(current_blocks)

        rng = random.Random(123)
        no_improve_count = 0
        max_no_improve = 600

        while time.time() < max_time and no_improve_count < max_no_improve:
            move_type = rng.random()

            if move_type < 0.40:
                if len(current_blocks) == 0:
                    no_improve_count += 1
                    continue
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                arr = block_t_arrays[b_idx]
                if not bool(np.any(cover_count[arr] == 1)):
                    cover_count[arr] -= 1
                    current_blocks.pop(i)
                    current_blocks_set.discard(b_idx)
                    no_improve_count = 0
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                else:
                    no_improve_count += 1

            elif move_type < 0.90:
                if len(current_blocks) == 0:
                    no_improve_count += 1
                    continue
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                old_arr = block_t_arrays[old_b]
                unique_ts = old_arr[cover_count[old_arr] == 1]
                if len(unique_ts) > 0:
                    cands = set(t_sub_to_k_arrays[unique_ts[0]].tolist())
                    for t_idx in unique_ts[1:]:
                        cands.intersection_update(set(t_sub_to_k_arrays[t_idx].tolist()))
                    cands.discard(old_b)
                    cands -= current_blocks_set
                else:
                    cands = set()
                    for _ in range(12):
                        c = rng.randint(0, n_k - 1)
                        if c not in current_blocks_set:
                            cands.add(c)
                            break
                if not cands:
                    no_improve_count += 1
                    continue
                new_b = rng.choice(list(cands))
                new_arr = block_t_arrays[new_b]
                cover_count[old_arr] -= 1
                cover_count[new_arr] += 1
                current_blocks[i] = new_b
                current_blocks_set.discard(old_b)
                current_blocks_set.add(new_b)
                no_improve_count = 0

                # Aggressive reduction after swap
                changed = True
                limit = 6
                while changed and limit > 0:
                    changed = False
                    limit -= 1
                    rng.shuffle(current_blocks)
                    for j in range(len(current_blocks)):
                        b_idx = current_blocks[j]
                        arr = block_t_arrays[b_idx]
                        if not bool(np.any(cover_count[arr] == 1)):
                            cover_count[arr] -= 1
                            current_blocks.pop(j)
                            current_blocks_set.discard(b_idx)
                            changed = True
                            if len(current_blocks) < best_len:
                                best_len = len(current_blocks)
                                best_blocks = list(current_blocks)
                            break
            else:
                no_improve_count += 1

        return best_blocks

    # Run initial greedy with "gain" strategy
    blocks_idx = greedy_cover(0, "gain")
    blocks_idx = reduce_blocks(blocks_idx)

    best_blocks = blocks_idx
    best_len = len(blocks_idx)

    # Try multiple strategies and seeds if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 12.0:
        b_idx = greedy_cover(42, "hardest")
        b_idx = reduce_blocks(b_idx)
        if b_idx and len(b_idx) < best_len:
            best_blocks = b_idx
            best_len = len(b_idx)

        remaining_time = deadline - time.time()
        if remaining_time > 8.0:
            seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
            for seed in seeds:
                if time.time() >= deadline - 6.0:
                    break
                b_idx = greedy_cover(seed, "gain")
                b_idx = reduce_blocks(b_idx)
                if b_idx and len(b_idx) < best_len:
                    best_blocks = b_idx
                    best_len = len(b_idx)

        remaining_time = deadline - time.time()
        if remaining_time > 6.0:
            seeds = [10, 20, 30, 50, 70, 90]
            for seed in seeds:
                if time.time() >= deadline - 5.0:
                    break
                b_idx = greedy_cover(seed, "hardest")
                b_idx = reduce_blocks(b_idx)
                if b_idx and len(b_idx) < best_len:
                    best_blocks = b_idx
                    best_len = len(b_idx)

    # Run Simulated Annealing on the best solution found
    if best_blocks and time.time() < deadline - 3.0:
        sa_time = min(30.0, deadline - time.time() - 3.0)
        if sa_time > 5.0:
            sa_result = simulated_annealing(best_blocks, time.time() + sa_time)
            if len(sa_result) < best_len:
                best_blocks = sa_result
                best_len = len(best_blocks)

    # Run Local Search on the best solution found
    if best_blocks and time.time() < deadline - 2.0:
        ls_time = min(15.0, deadline - time.time() - 2.0)
        if ls_time > 3.0:
            ls_result = local_search_swap(best_blocks, time.time() + ls_time)
            if len(ls_result) < best_len:
                best_blocks = ls_result
                best_len = len(best_blocks)

    # Final reduction pass (exact redundancy polish)
    if best_blocks and time.time() < deadline - 1.0:
        best_blocks = reduce_blocks(best_blocks)

    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
