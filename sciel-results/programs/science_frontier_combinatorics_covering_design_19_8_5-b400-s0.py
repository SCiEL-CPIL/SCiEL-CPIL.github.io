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

            # With some probability, try 2-for-1
            if len(blocks) >= 2 and random.random() < 0.1:
                i, j = random.sample(range(len(blocks)), 2)
                ts_i = block_ts_indices[i]
                ts_j = block_ts_indices[j]
                
                # Find subsets that will be uncovered if we remove i and j
                # These are subsets in (ts_i U ts_j) with count <= 2
                # And specifically:
                # - In ts_i, count == 1 (unique to i)
                # - In ts_j, count == 1 (unique to j)
                # - In ts_i AND ts_j, count == 2 (unique to pair)
                
                # Efficient way:
                # Create a temporary view of cover_count? No, just check.
                # unique_to_pair = []
                # for x in np.unique(np.concatenate([ts_i, ts_j])):
                #    if cover_count[x] == 1 and x in ts_i: unique_to_pair.append(x)
                #    elif cover_count[x] == 1 and x in ts_j: unique_to_pair.append(x)
                #    elif cover_count[x] == 2 and x in ts_i and x in ts_j: unique_to_pair.append(x)
                
                # This loop is slow in Python. Use NumPy.
                # We need to know which subsets in ts_i are unique to i (count 1)
                # Which in ts_j are unique to j (count 1)
                # Which in intersection are unique to pair (count 2)
                
                # Let's approximate:
                # If we remove i and j, the "risk" subsets are those with count 1 in i, 1 in j, or 2 in both.
                # Let's just gather candidates for replacement.
                
                # Find intersection of ts_i and ts_j
                # This is O(N_i + N_j).
                # Set intersection is fast.
                set_i = set(ts_i)
                set_j = set(ts_j)
                inter = set_i.intersection(set_j)
                
                # Subsets unique to i (in i, not in j, count 1)
                # Subsets unique to j (in j, not in i, count 1)
                # Subsets in inter with count 2
                
                # We need a block k that covers ALL these subsets.
                # Let U be the set of these subsets.
                # If |U| > C(k,t), skip.
                
                # Construct U
                U = []
                # Unique to i
                for x in ts_i:
                    if x not in set_j and cover_count[x] == 1:
                        U.append(x)
                # Unique to j
                for x in ts_j:
                    if x not in set_i and cover_count[x] == 1:
                        U.append(x)
                # Unique to pair
                for x in inter:
                    if cover_count[x] == 2:
                        U.append(x)
                
                if len(U) > C(k, t):
                    continue
                
                # Find blocks covering U
                if not U:
                    # Can remove both?
                    # If no unique subsets, both are redundant?
                    # If U is empty, it means all subsets in i and j are covered by other blocks.
                    # So we can remove i and j.
                    # But we need to update cover_count.
                    np.subtract.at(cover_count, ts_i, 1)
                    np.subtract.at(cover_count, ts_j, 1)
                    # Remove i and j
                    # Be careful with indices
                    if i > j:
                        i, j = j, i
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    blocks.pop(j)
                    block_ts_indices.pop(j)
                    # Update sol_block_set
                    idx_i = get_block_idx(blocks[i]) if i < len(blocks) else -1
                    # This index shifting is tricky. Let's just rebuild sol_block_set later or use a different structure.
                    # For now, just count improvements.
                    improved_count += 2
                    # Rebuild sol_block_set for safety
                    sol_block_set.clear()
                    for b in blocks:
                        idx = get_block_idx(b)
                        if idx >= 0:
                            sol_block_set.add(idx)
                    continue

                # Find intersection of ts_to_blocks for all x in U
                # Start with all blocks, intersect with ts_to_blocks[x]
                # This is expensive if U is large.
                # If U is small, it's fine.
                if len(U) > 10:
                    continue # Heuristic: too many constraints, unlikely to find a block
                
                candidates = set(range(num_k_blocks))
                for x in U:
                    candidates.intersection_update(ts_to_blocks[x])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Check if any candidate is valid (not i, not j)
                valid_cands = [c for c in candidates if c != get_block_idx(blocks[i]) and c != get_block_idx(blocks[j])]
                if not valid_cands:
                    continue
                
                new_idx = random.choice(valid_cands)
                new_ts = block_to_ts_np[new_idx]
                
                # Check if new_ts covers U
                # Since candidates came from intersection, it should.
                
                # Perform swap
                # Remove i, j
                # Add new_idx
                # Update cover_count
                np.subtract.at(cover_count, ts_i, 1)
                np.subtract.at(cover_count, ts_j, 1)
                np.add.at(cover_count, new_ts, 1)
                
                # Check validity: min(cover_count) >= 1
                if np.min(cover_count) >= 1:
                    # Update blocks list
                    # Remove i, j
                    if i > j:
                        i, j = j, i
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    blocks.pop(j)
                    block_ts_indices.pop(j)
                    
                    # Add new block
                    blocks.append(list(all_k_blocks[new_idx]))
                    block_ts_indices.append(new_ts)
                    
                    # Update sol_block_set
                    sol_block_set.discard(get_block_idx(blocks[i]) if i < len(blocks) else -1) # Approximate
                    sol_block_set.discard(get_block_idx(blocks[j]) if j < len(blocks) else -1)
                    sol_block_set.add(new_idx)
                    
                    improved_count += 1
                else:
                    # Rollback
                    np.subtract.at(cover_count, new_ts, 1)
                    np.add.at(cover_count, ts_i, 1)
                    np.add.at(cover_count, ts_j, 1)
            else:
                # Standard 1-exchange
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

        return blocks, improved_count

    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)

    # Run multiple randomized greedy
    num_runs = 150
    for i in range(1, num_runs):
        if time.time() > deadline - 40.0:
            break

        explore = 0.0 if i < 50 else (0.1 if i < 100 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Phase 2: Iterated Local Search
    for _ in range(300):
        if time.time() > deadline - 5.0:
            break
        
        best_solution, improved = local_search(best_solution, max_iters=2000)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
        
        if improved == 0:
            # Try a new greedy run
            if time.time() < deadline - 10.0:
                sol = run_greedy(random.randint(1000, 10000), explore_prob=0.1)
                if sol is not None and len(sol) < best_len:
                    best_len = len(sol)
                    best_solution = sol

    return best_solution
