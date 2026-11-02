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
        rng = random.Random(seed)
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

            if explore_prob > 0 and rng.random() < explore_prob:
                # Explore: pick from top 5% or gain >= max_gain - 3
                threshold = max_gain - 3
                explore_cands = np.where((gain >= threshold) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[rng.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            is_uncovered[newly_covered] = False

            # Vectorized gain update
            if len(newly_covered) > 0:
                all_blocks_to_decrement = []
                for ts_idx in newly_covered:
                    all_blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                if all_blocks_to_decrement:
                    counts = np.bincount(all_blocks_to_decrement, minlength=num_k_blocks)
                    gain -= counts
                    gain[is_selected] = -1000

        return sol

    def local_search(blocks, max_iters=3000):
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

        # Phase B: 1-exchange and 2-for-1 local search
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 3.0:
                break
            if len(blocks) == 0:
                break

            # Try 1-exchange first
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
                # Try 2-for-1: remove block i and add two new blocks
                if len(blocks) >= 2:
                    j = random.randint(0, len(blocks) - 1)
                    if j == i:
                        j = (j + 1) % len(blocks)
                    
                    ts_arr_j = block_ts_indices[j]
                    unique_mask_j = cover_count[ts_arr_j] == 1
                    unique_ts_j = ts_arr_j[unique_mask_j]
                    
                    if len(unique_ts_j) == 0:
                        # Block j is fully redundant, just remove it
                        np.subtract.at(cover_count, ts_arr_j, 1)
                        blocks.pop(j)
                        block_ts_indices.pop(j)
                        if j < i:
                            i -= 1
                        improved_count += 1
                        continue
                    
                    # Find blocks that cover unique_ts of both i and j
                    all_unique = np.concatenate([unique_ts, unique_ts_j])
                    all_unique = np.unique(all_unique)
                    
                    if len(all_unique) > 0:
                        cand_set = set(ts_to_blocks[int(all_unique[0])])
                        for ts_idx in all_unique[1:]:
                            cand_set.intersection_update(ts_to_blocks[int(ts_idx)])
                            if not cand_set:
                                break
                        
                        if cand_set:
                            valid_cands = [c for c in cand_set if c not in sol_block_set]
                            if valid_cands:
                                new_idx1 = random.choice(valid_cands)
                                remaining = [c for c in valid_cands if c != new_idx1]
                                if remaining:
                                    new_idx2 = random.choice(remaining)
                                else:
                                    # Need two different blocks, pick another if possible
                                    new_idx2 = new_idx1  # Will check validity below
                                    
                                if new_idx1 != new_idx2:
                                    ts_arr_n1 = block_to_ts_np[new_idx1]
                                    ts_arr_n2 = block_to_ts_np[new_idx2]
                                    
                                    np.subtract.at(cover_count, ts_arr_i, 1)
                                    np.subtract.at(cover_count, ts_arr_j, 1)
                                    np.add.at(cover_count, ts_arr_n1, 1)
                                    np.add.at(cover_count, ts_arr_n2, 1)
                                    
                                    if np.min(cover_count) >= 1:
                                        # Remove j first (higher index) to keep i stable
                                        blocks.pop(j)
                                        block_ts_indices.pop(j)
                                        if j < i:
                                            i -= 1
                                        blocks.pop(i)
                                        block_ts_indices.pop(i)
                                        
                                        blocks.append(list(all_k_blocks[new_idx1]))
                                        block_ts_indices.append(ts_arr_n1)
                                        blocks.append(list(all_k_blocks[new_idx2]))
                                        block_ts_indices.append(ts_arr_n2)
                                        
                                        sol_block_set.discard(get_block_idx(blocks[-2]))
                                        sol_block_set.discard(get_block_idx(blocks[-1]))
                                        sol_block_set.add(new_idx1)
                                        sol_block_set.add(new_idx2)
                                        improved_count += 1
                                    else:
                                        np.subtract.at(cover_count, ts_arr_n1, 1)
                                        np.subtract.at(cover_count, ts_arr_n2, 1)
                                        np.add.at(cover_count, ts_arr_i, 1)
                                        np.add.at(cover_count, ts_arr_j, 1)
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

        return blocks, improved_count

    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)
    
    # Run local search on initial solution
    best_solution, _ = local_search(best_solution, max_iters=2000)
    best_len = len(best_solution)

    # Run multiple randomized greedy with local search
    num_runs = 100
    for i in range(1, num_runs):
        if time.time() > deadline - 60.0:
            break

        explore = 0.0 if i < 30 else (0.1 if i < 60 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None:
            sol, _ = local_search(sol, max_iters=1000)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol

    # Phase 2: Iterated Local Search with valid perturbations
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Valid perturbation: swap one block with another that covers its unique t-subsets
        if len(best_solution) > 1:
            # Try to swap a block with another that maintains coverage
            i = random.randint(0, len(best_solution) - 1)
            block_i = best_solution[i]
            block_idx_i = get_block_idx(block_i)
            
            if block_idx_i >= 0:
                ts_arr_i = block_to_ts_np[block_idx_i]
                
                # Compute coverage counts
                cover_count = np.zeros(num_t_subsets, dtype=np.int32)
                for b in best_solution:
                    idx = get_block_idx(b)
                    if idx >= 0:
                        np.add.at(cover_count, block_to_ts_np[idx], 1)
                
                # Find unique t-subsets of block i
                unique_mask = cover_count[ts_arr_i] == 1
                unique_ts = ts_arr_i[unique_mask]
                
                if len(unique_ts) == 0:
                    # Block i is redundant, just remove it
                    best_solution.pop(i)
                    best_len -= 1
                    continue
                
                # Find candidate blocks that cover all unique t-subsets
                candidates = set(ts_to_blocks[int(unique_ts[0])])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                    if not candidates:
                        break
                
                if candidates:
                    valid_cands = [c for c in candidates if c != block_idx_i]
                    if valid_cands:
                        new_block_idx = random.choice(valid_cands)
                        new_block = list(all_k_blocks[new_block_idx])
                        
                        # Verify the swap maintains coverage
                        cover_count[ts_arr_i] -= 1
                        np.add.at(cover_count, block_to_ts_np[new_block_idx], 1)
                        
                        if np.min(cover_count) >= 1:
                            best_solution[i] = new_block
                        else:
                            # Revert
                            cover_count[ts_arr_i] += 1
                            np.subtract.at(cover_count, block_to_ts_np[new_block_idx], 1)
        
        # Run local search
        best_solution, improved = local_search(best_solution, max_iters=1500)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
        
        if improved == 0:
            # Try a new greedy run
            if time.time() < deadline - 10.0:
                sol = run_greedy(random.randint(1000, 10000), explore_prob=0.15)
                if sol is not None:
                    sol, _ = local_search(sol, max_iters=800)
                    if len(sol) < best_len:
                        best_len = len(sol)
                        best_solution = sol

    return best_solution
