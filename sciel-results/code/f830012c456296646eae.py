import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin for 150s limit

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

    def run_greedy(seed, explore_prob=0.0):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 5.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            if explore_prob > 0 and random.random() < explore_prob:
                explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[random.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            is_uncovered[newly_covered] = False

            if len(newly_covered) > 0:
                all_block_indices = np.concatenate(
                    [ts_to_blocks_np[int(ts_idx)] for ts_idx in newly_covered]
                )
                if len(all_block_indices) > 0:
                    unique_blocks, counts = np.unique(all_block_indices, return_counts=True)
                    gain[unique_blocks] -= counts
                    gain[is_selected] = -1000

        return sol

    def run_t_packing_greedy(seed):
        """Greedy that starts with a t-packing scaffold."""
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        # Phase 1: Build t-packing (blocks with disjoint t-subsets)
        # Use greedy: pick block with max new coverage, prefer blocks with low overlap
        packing_gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        covered_ts_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            # Score = new coverage - overlap penalty
            max_packing_gain = int(np.max(packing_gain))
            if max_packing_gain <= 0:
                break
            
            candidates = np.where((packing_gain == max_packing_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
            
            b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            packing_gain[b_idx] = -1
            
            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
            
            is_uncovered[newly_covered] = False
            covered_ts_count[bts] += 1
            
            # Update packing gain: penalize blocks that overlap with already covered t-subsets
            if len(newly_covered) > 0:
                all_block_indices = np.concatenate(
                    [ts_to_blocks_np[int(ts_idx)] for ts_idx in newly_covered]
                )
                if len(all_block_indices) > 0:
                    unique_blocks, counts = np.unique(all_block_indices, return_counts=True)
                    packing_gain[unique_blocks] -= counts * 2  # Stronger penalty for packing
                    packing_gain[is_selected] = -1000
                    gain[unique_blocks] -= counts
                    gain[is_selected] = -1000

        # Phase 2: Continue with standard greedy to cover remaining
        while True:
            if time.time() > deadline - 5.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            is_uncovered[newly_covered] = False

            if len(newly_covered) > 0:
                all_block_indices = np.concatenate(
                    [ts_to_blocks_np[int(ts_idx)] for ts_idx in newly_covered]
                )
                if len(all_block_indices) > 0:
                    unique_blocks, counts = np.unique(all_block_indices, return_counts=True)
                    gain[unique_blocks] -= counts
                    gain[is_selected] = -1000

        return sol

    def local_search(blocks, max_iters=3000):
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0

        # Build cover count and block-to-ts mapping for current solution
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

            # Find t-subsets uniquely covered by block i
            unique_mask = cover_count[ts_arr_i] == 1
            unique_ts = ts_arr_i[unique_mask]

            if len(unique_ts) == 0:
                # Block is redundant, remove it
                np.subtract.at(cover_count, ts_arr_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                improved_count += 1
                continue

            # Find candidate blocks that cover all unique_ts
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
                # Revert
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_i, 1)

        # Phase C: 2-exchange local search (replace 2 blocks with 2 new blocks)
        iters2 = 0
        while iters2 < max_iters // 3:
            iters2 += 1
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

            # Find t-subsets uniquely covered by blocks i and j
            temp_cover = cover_count.copy()
            np.subtract.at(temp_cover, ts_arr_i, 1)
            np.subtract.at(temp_cover, ts_arr_j, 1)
            
            if np.min(temp_cover) < 0:
                # Can't remove both
                continue

            # Find t-subsets that need coverage (uniquely covered by i or j)
            unique_to_ij = (temp_cover == 0) & (cover_count == 1)
            unique_ts_set = np.where(unique_to_ij)[0]
            
            if len(unique_ts_set) == 0:
                # Both are redundant
                np.subtract.at(cover_count, ts_arr_i, 1)
                np.subtract.at(cover_count, ts_arr_j, 1)
                # Remove j first (higher index) then i
                blocks.pop(j)
                block_ts_indices.pop(j)
                if i > j:
                    i -= 1
                blocks.pop(i)
                block_ts_indices.pop(i)
                improved_count += 2
                continue

            # Find candidate blocks that cover all unique_ts_set
            candidates = set(ts_to_blocks[int(unique_ts_set[0])])
            for ts_idx in unique_ts_set[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break

            if not candidates:
                continue

            valid_candidates = [c for c in candidates if c not in sol_block_set and c != get_block_idx(blocks[i]) and c != get_block_idx(blocks[j])]
            if len(valid_candidates) < 2:
                continue

            # Pick 2 random candidates
            chosen = random.sample(valid_candidates, 2)
            new_ts_arr_1 = block_to_ts_np[chosen[0]]
            new_ts_arr_2 = block_to_ts_np[chosen[1]]

            # Check if swap is valid
            np.subtract.at(cover_count, ts_arr_i, 1)
            np.subtract.at(cover_count, ts_arr_j, 1)
            np.add.at(cover_count, new_ts_arr_1, 1)
            np.add.at(cover_count, new_ts_arr_2, 1)

            if np.min(cover_count) >= 1:
                blocks[i] = list(all_k_blocks[chosen[0]])
                blocks[j] = list(all_k_blocks[chosen[1]])
                block_ts_indices[i] = new_ts_arr_1
                block_ts_indices[j] = new_ts_arr_2
                old_idx_i = get_block_idx(blocks[i])
                old_idx_j = get_block_idx(blocks[j])
                if old_idx_i >= 0:
                    sol_block_set.discard(old_idx_i)
                if old_idx_j >= 0:
                    sol_block_set.discard(old_idx_j)
                sol_block_set.add(chosen[0])
                sol_block_set.add(chosen[1])
                improved_count += 2
            else:
                # Revert
                np.subtract.at(cover_count, new_ts_arr_1, 1)
                np.subtract.at(cover_count, new_ts_arr_2, 1)
                np.add.at(cover_count, ts_arr_i, 1)
                np.add.at(cover_count, ts_arr_j, 1)

        return blocks, improved_count

    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)

    # Run t-packing greedy variants
    for i in range(30):
        if time.time() > deadline - 60.0:
            break
        sol = run_t_packing_greedy(1000 + i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Run multiple randomized greedy with varying exploration
    num_runs = 200
    for i in range(1, num_runs):
        if time.time() > deadline - 40.0:
            break

        explore = 0.0 if i < 60 else (0.1 if i < 120 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Phase 2: Iterative local search
    for _ in range(800):
        if time.time() > deadline - 5.0:
            break
        best_solution, improved = local_search(best_solution, max_iters=2000)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
        if improved == 0 and time.time() > deadline - 15.0:
            break

    return best_solution
