import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]

    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    block_to_ts_np = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_greedy(seed, explore_prob=0.0, use_weighted=False):
        rng = random.Random(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        block_freq = np.zeros(num_k_blocks, dtype=np.int32)

        while True:
            if time.time() > deadline - 5.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            if explore_prob > 0 and rng.random() < explore_prob:
                threshold = max(0, max_gain - 5)
                explore_cands = np.where((gain >= threshold) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    if use_weighted and len(explore_cands) > 1:
                        weights = gain[explore_cands] + block_freq[explore_cands] * 0.1
                        weights = weights / np.sum(weights)
                        b_idx = int(explore_cands[np.random.choice(len(explore_cands), p=weights)])
                    else:
                        b_idx = int(explore_cands[rng.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
            else:
                if use_weighted and len(candidates) > 1:
                    weights = gain[candidates] + block_freq[candidates] * 0.1
                    weights = weights / np.sum(weights)
                    b_idx = int(candidates[np.random.choice(len(candidates), p=weights)])
                else:
                    b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            block_freq[b_idx] += 1

            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            is_uncovered[newly_covered] = False

            if len(newly_covered) > 0:
                all_blocks_to_decrement = []
                for ts_idx in newly_covered:
                    all_blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                if all_blocks_to_decrement:
                    counts = np.bincount(all_blocks_to_decrement, minlength=num_k_blocks)
                    gain -= counts
                    gain[is_selected] = -1000

        return sol

    def local_search(blocks, max_iters=3000, use_2exchange=False):
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0

        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts_np[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                arr = np.array(ts_list, dtype=np.int32)
                block_ts_indices.append(arr)

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for ts_arr in block_ts_indices:
            np.add.at(cover_count, ts_arr, 1)

        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_set.add(idx)

        improved_count = 0

        # Phase A: Iteratively remove redundant blocks
        for _pass in range(8):
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks)):
                    if time.time() > deadline - 3.0:
                        return blocks, improved_count
                    ts_arr = block_ts_indices[i]
                    if np.all(cover_count[ts_arr] >= 2):
                        np.subtract.at(cover_count, ts_arr, 1)
                        blocks.pop(i)
                        block_ts_indices.pop(i)
                        if i < len(blocks):
                            idx = get_block_idx(blocks[i])
                        else:
                            idx = -1
                        if idx >= 0:
                            sol_block_set.discard(idx)
                        improved_count += 1
                        changed = True
                        break

        # Phase B: 1-exchange local search
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 3.0:
                break
            if len(blocks) == 0:
                break

            i = random.randint(0, len(blocks) - 1)
            ts_arr_i = block_ts_indices[i]

            unique_mask = cover_count[ts_arr_i] == 1
            unique_ts = ts_arr_i[unique_mask]

            if len(unique_ts) == 0:
                np.subtract.at(cover_count, ts_arr_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                improved_count += 1
                continue

            candidates = set(ts_to_blocks[int(unique_ts[0])])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break

            if not candidates:
                continue

            valid_candidates = [c for c in candidates if c not in sol_block_set]
            if not valid_candidates:
                continue

            new_block_idx = random.choice(valid_candidates)
            new_ts_arr = block_to_ts_np[new_block_idx]

            np.subtract.at(cover_count, ts_arr_i, 1)
            np.add.at(cover_count, new_ts_arr, 1)

            if np.min(cover_count) >= 1:
                blocks[i] = list(all_k_blocks[new_block_idx])
                block_ts_indices[i] = new_ts_arr
                old_idx = get_block_idx(blocks[i])
                if old_idx >= 0:
                    sol_block_set.discard(old_idx)
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_i, 1)

        # Phase C: 2-exchange if enabled
        if use_2exchange:
            two_ex_iters = 0
            while two_ex_iters < max_iters // 2:
                two_ex_iters += 1
                if time.time() > deadline - 3.0:
                    break
                if len(blocks) < 2:
                    break

                i = random.randint(0, len(blocks) - 1)
                j = random.randint(0, len(blocks) - 1)
                if i == j:
                    continue

                ts_arr_i = block_ts_indices[i]
                ts_arr_j = block_ts_indices[j]

                # Find unique t-subsets in block i
                unique_mask_i = cover_count[ts_arr_i] == 1
                unique_ts_i = ts_arr_i[unique_mask_i]

                if len(unique_ts_i) == 0:
                    # Just remove block i
                    np.subtract.at(cover_count, ts_arr_i, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    idx = get_block_idx(blocks[i]) if i < len(blocks) else -1
                    if idx >= 0:
                        sol_block_set.discard(idx)
                    improved_count += 1
                    continue

                # Find candidates that cover unique_ts_i
                candidates = set(ts_to_blocks[int(unique_ts_i[0])])
                for ts_idx in unique_ts_i[1:]:
                    candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                    if not candidates:
                        break

                if not candidates:
                    continue

                # For each candidate, check if we can also improve block j
                for new_i_idx in list(candidates):
                    if new_i_idx in sol_block_set:
                        continue
                    new_ts_arr_i = block_to_ts_np[new_i_idx]

                    # Simulate removing block i and adding new_i
                    temp_cover = cover_count.copy()
                    np.subtract.at(temp_cover, ts_arr_i, 1)
                    np.add.at(temp_cover, new_ts_arr_i, 1)

                    if np.min(temp_cover) < 1:
                        continue

                    # Now try to improve block j
                    ts_arr_j = block_ts_indices[j]
                    unique_mask_j = temp_cover[ts_arr_j] == 1
                    unique_ts_j = ts_arr_j[unique_mask_j]

                    if len(unique_ts_j) == 0:
                        # Can remove block j
                        np.subtract.at(temp_cover, ts_arr_j, 1)
                        if np.min(temp_cover) >= 1:
                            # Apply both changes
                            np.subtract.at(cover_count, ts_arr_i, 1)
                            np.add.at(cover_count, new_ts_arr_i, 1)
                            np.subtract.at(cover_count, ts_arr_j, 1)
                            blocks[i] = list(all_k_blocks[new_i_idx])
                            block_ts_indices[i] = new_ts_arr_i
                            blocks.pop(j)
                            block_ts_indices.pop(j)
                            if j < len(blocks):
                                idx = get_block_idx(blocks[j])
                            else:
                                idx = -1
                            if idx >= 0:
                                sol_block_set.discard(idx)
                            old_idx_i = get_block_idx(blocks[i])
                            if old_idx_i >= 0:
                                sol_block_set.discard(old_idx_i)
                            sol_block_set.add(new_i_idx)
                            improved_count += 1
                        break
                    else:
                        # Find candidate for block j
                        candidates_j = set(ts_to_blocks[int(unique_ts_j[0])])
                        for ts_idx in unique_ts_j[1:]:
                            candidates_j.intersection_update(ts_to_blocks[int(ts_idx)])
                            if not candidates_j:
                                break

                        if not candidates_j:
                            continue

                        for new_j_idx in list(candidates_j):
                            if new_j_idx in sol_block_set or new_j_idx == new_i_idx:
                                continue
                            new_ts_arr_j = block_to_ts_np[new_j_idx]

                            temp_cover2 = temp_cover.copy()
                            np.subtract.at(temp_cover2, ts_arr_j, 1)
                            np.add.at(temp_cover2, new_ts_arr_j, 1)

                            if np.min(temp_cover2) >= 1:
                                # Apply all changes
                                np.subtract.at(cover_count, ts_arr_i, 1)
                                np.add.at(cover_count, new_ts_arr_i, 1)
                                np.subtract.at(cover_count, ts_arr_j, 1)
                                np.add.at(cover_count, new_ts_arr_j, 1)
                                blocks[i] = list(all_k_blocks[new_i_idx])
                                block_ts_indices[i] = new_ts_arr_i
                                blocks[j] = list(all_k_blocks[new_j_idx])
                                block_ts_indices[j] = new_ts_arr_j
                                old_idx_i = get_block_idx(blocks[i])
                                if old_idx_i >= 0:
                                    sol_block_set.discard(old_idx_i)
                                old_idx_j = get_block_idx(blocks[j])
                                if old_idx_j >= 0:
                                    sol_block_set.discard(old_idx_j)
                                sol_block_set.add(new_i_idx)
                                sol_block_set.add(new_j_idx)
                                improved_count += 1
                                break
                        break

        # Phase D: Final cleanup
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    return blocks, improved_count
                ts_arr = block_ts_indices[i]
                if np.all(cover_count[ts_arr] >= 2):
                    np.subtract.at(cover_count, ts_arr, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    if i < len(blocks):
                        idx = get_block_idx(blocks[i])
                    else:
                        idx = -1
                    if idx >= 0:
                        sol_block_set.discard(idx)
                    improved_count += 1
                    changed = True
                    break

        return blocks, improved_count

    # Initial deterministic greedy
    best_solution = run_greedy(0, use_weighted=True)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)

    # Run local search on initial solution
    best_solution, _ = local_search(best_solution, max_iters=4000, use_2exchange=True)
    best_len = len(best_solution)

    # Run multiple randomized greedy with local search
    num_runs = 100
    for i in range(1, num_runs):
        if time.time() > deadline - 60.0:
            break

        explore = 0.0 if i < 15 else (0.08 if i < 40 else 0.15)
        use_w = (i % 3 == 0)
        sol = run_greedy(i, explore_prob=explore, use_weighted=use_w)
        if sol is not None:
            use_2ex = (i % 4 == 0)
            sol, _ = local_search(sol, max_iters=2000, use_2exchange=use_2ex)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol

    # Phase 2: Iterated Local Search
    no_improve_count = 0
    for _ in range(500):
        if time.time() > deadline - 5.0:
            break

        # Perturb
        if len(best_solution) > 2:
            # Random 2-exchange perturbation
            i = random.randint(0, len(best_solution) - 1)
            j = random.randint(0, len(best_solution) - 1)
            if i != j:
                block_i = best_solution[i]
                block_j = best_solution[j]
                idx_i = get_block_idx(block_i)
                idx_j = get_block_idx(block_j)

                if idx_i >= 0 and idx_j >= 0:
                    ts_arr_i = block_to_ts_np[idx_i]
                    ts_arr_j = block_to_ts_np[idx_j]

                    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
                    for b in best_solution:
                        idx = get_block_idx(b)
                        if idx >= 0:
                            np.add.at(cover_count, block_to_ts_np[idx], 1)

                    # Try to find alternative blocks
                    unique_mask_i = cover_count[ts_arr_i] == 1
                    unique_ts_i = ts_arr_i[unique_mask_i]

                    if len(unique_ts_i) > 0:
                        candidates_i = set(ts_to_blocks[int(unique_ts_i[0])])
                        for ts_idx in unique_ts_i[1:]:
                            candidates_i.intersection_update(ts_to_blocks[int(ts_idx)])
                            if not candidates_i:
                                break

                        valid_cands_i = [c for c in candidates_i if c != idx_i]
                        if valid_cands_i:
                            new_i_idx = random.choice(valid_cands_i)
                            new_ts_arr_i = block_to_ts_np[new_i_idx]

                            temp_cover = cover_count.copy()
                            np.subtract.at(temp_cover, ts_arr_i, 1)
                            np.add.at(temp_cover, new_ts_arr_i, 1)

                            if np.min(temp_cover) >= 1:
                                best_solution[i] = list(all_k_blocks[new_i_idx])

        best_solution, improved = local_search(best_solution, max_iters=2500, use_2exchange=True)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= 15 and time.time() < deadline - 15.0:
            sol = run_greedy(random.randint(5000, 15000), explore_prob=0.12, use_weighted=True)
            if sol is not None:
                sol, _ = local_search(sol, max_iters=1500, use_2exchange=True)
                if len(sol) < best_len:
                    best_len = len(sol)
                    best_solution = sol
                    no_improve_count = 0

    # Final polish
    if time.time() < deadline - 10.0:
        best_solution, _ = local_search(best_solution, max_iters=6000, use_2exchange=True)
        best_len = len(best_solution)

    return best_solution
