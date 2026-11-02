import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin for 150s limit

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_t = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}

    all_kblocks = list(itertools.combinations(range(v), k))
    num_k = len(all_kblocks)

    # Incidence: block_to_ts[b] = array of t-subset indices covered by block b
    # ts_to_blocks[ts] = list of block indices covering t-subset ts
    t_in_k = list(itertools.combinations(range(k), t))
    num_t_in_k = len(t_in_k)

    block_to_ts = np.zeros((num_k, num_t_in_k), dtype=np.int32)
    ts_to_blocks = [[] for _ in range(num_t)]

    for i, block in enumerate(all_kblocks):
        for j, t_indices in enumerate(t_in_k):
            global_ts = tuple(sorted(block[x] for x in t_indices))
            idx = tset_to_idx[global_ts]
            block_to_ts[i, j] = idx
            ts_to_blocks[idx].append(i)

    block_to_idx = {block: i for i, block in enumerate(all_kblocks)}

    # 2. Fast incremental greedy
    def run_greedy(randomized=False, seed=None):
        if seed is not None:
            random.seed(seed)

        gain = np.full(num_k, num_t_in_k, dtype=np.int32)
        is_uncovered = np.full(num_t, True, dtype=bool)
        is_selected = np.zeros(num_k, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 20.0:
                return None

            if randomized:
                # Find max gain among unselected
                max_gain = 0
                for i in range(num_k):
                    if not is_selected[i] and gain[i] > max_gain:
                        max_gain = gain[i]
                if max_gain <= 0:
                    break
                # Collect candidates with max gain
                candidates = [i for i in range(num_k) if not is_selected[i] and gain[i] == max_gain]
                b_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                # Vectorized max
                temp_gain = gain.copy()
                temp_gain[is_selected] = -1
                max_gain = int(np.max(temp_gain))
                if max_gain <= 0:
                    break
                b_idx = int(np.argmax(temp_gain))

            sol.append(b_idx)
            is_selected[b_idx] = True

            # Update gains incrementally
            tids = block_to_ts[b_idx]
            for ts_idx in tids:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb in ts_to_blocks[ts_idx]:
                        if not is_selected[nb]:
                            gain[nb] -= 1

        return sol

    # 3. Polish: remove redundant blocks (vectorized)
    def polish(sol_indices):
        if not sol_indices:
            return sol_indices

        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in sol_indices:
            cover_count[block_to_ts[b]] += 1

        new_sol = []
        for b in sol_indices:
            if np.min(cover_count[block_to_ts[b]]) > 1:
                cover_count[block_to_ts[b]] -= 1
            else:
                new_sol.append(b)
        return new_sol

    # 4. 1-exchange swap local search
    def swap_search(sol_indices):
        if len(sol_indices) < 2:
            return sol_indices, False

        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in sol_indices:
            cover_count[block_to_ts[b]] += 1

        improved = False
        sol_list = list(sol_indices)

        for _ in range(20):
            if time.time() > deadline - 20.0:
                break
            found = False
            for i in range(len(sol_list)):
                b = sol_list[i]
                tids = block_to_ts[b]
                # Find t-subsets uniquely covered by b
                unique_mask = cover_count[tids] == 1
                if not np.any(unique_mask):
                    continue
                unique_ts = tids[unique_mask]

                # Find candidates covering all unique_ts
                candidates = set(ts_to_blocks[int(unique_ts[0])])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                    if not candidates:
                        break

                candidates.discard(b)
                if not candidates:
                    continue

                # Pick candidate that maximizes coverage of currently-overcovered t-sets
                best_c = None
                best_score = -1
                for c in list(candidates)[:50]:  # Limit candidates
                    # Score: how many t-sets does c cover that have cover_count > 1?
                    c_tids = block_to_ts[c]
                    score = int(np.sum(cover_count[c_tids] > 1))
                    if score > best_score:
                        best_score = score
                        best_c = c

                if best_c is not None:
                    # Apply swap
                    cover_count[block_to_ts[b]] -= 1
                    cover_count[block_to_ts[best_c]] += 1
                    sol_list[i] = best_c
                    improved = True
                    found = True
                    break
            if not found:
                break

        return sol_list, improved

    # 5. Pairwise merge
    def pairwise_merge(sol_indices):
        if len(sol_indices) < 2:
            return sol_indices, False
        if time.time() > deadline - 20.0:
            return sol_indices, False

        n = len(sol_indices)
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in sol_indices:
            cover_count[block_to_ts[b]] += 1

        # For each block, find uniquely covered t-subsets
        unique_sets = []
        for b in sol_indices:
            tids = block_to_ts[b]
            unique = set(tids[cover_count[tids] == 1].tolist())
            unique_sets.append(unique)

        new_sol = list(sol_indices)
        changed = False

        for i in range(n):
            if time.time() > deadline - 15.0:
                break
            for j in range(i + 1, n):
                if new_sol[i] is None or new_sol[j] is None:
                    continue
                union_unique = unique_sets[i] | unique_sets[j]
                if not union_unique:
                    continue

                # Check if a k-block can contain all elements from these t-subsets
                all_elements = set()
                for tid in union_unique:
                    all_elements.update(all_tsets[tid])

                if len(all_elements) > k:
                    continue

                remaining = set(range(v)) - all_elements
                need = k - len(all_elements)
                if need < 0:
                    continue

                sorted_all = sorted(all_elements)
                sorted_rem = sorted(remaining)
                if need == 0:
                    new_block = sorted_all
                else:
                    new_block = sorted(sorted_all + sorted_rem[:need])

                new_block_tuple = tuple(new_block)
                if new_block_tuple in block_to_idx:
                    new_idx = block_to_idx[new_block_tuple]
                    new_tids = set(block_to_ts[new_idx].tolist())
                    if union_unique.issubset(new_tids):
                        new_sol[i] = None
                        new_sol[j] = None
                        new_sol.append(new_idx)
                        changed = True
                        break
            if changed:
                break

        if changed:
            new_sol = [b for b in new_sol if b is not None]

        return new_sol, changed

    # 6. WalkSAT-style local search
    def walksat(sol_indices, noise, max_steps, time_limit):
        included = np.zeros(num_k, dtype=bool)
        for b in sol_indices:
            included[b] = True

        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in range(num_k):
            if included[i]:
                cover_count[block_to_ts[i]] += 1

        num_uncovered = int(np.sum(cover_count == 0))
        best_num_uncovered = num_uncovered
        best_included = included.copy()
        best_count = int(np.sum(included))

        step = 0
        while step < max_steps and time.time() < time_limit:
            step += 1

            if random.random() < noise:
                i = random.randint(0, num_k - 1)
            else:
                # Compute delta for all blocks
                delta = np.zeros(num_k, dtype=np.int32)
                inc_mask = included
                if np.any(inc_mask):
                    inc_tids = block_to_ts[inc_mask]
                    delta[inc_mask] = np.sum(cover_count[inc_tids] == 1, axis=1)
                not_inc_mask = ~included
                if np.any(not_inc_mask):
                    not_inc_tids = block_to_ts[not_inc_mask]
                    delta[not_inc_mask] = -np.sum(cover_count[not_inc_tids] == 0, axis=1)
                i = int(np.argmin(delta))

            tids_i = block_to_ts[i]
            if included[i]:
                before_one = int(np.sum(cover_count[tids_i] == 1))
                cover_count[tids_i] -= 1
                included[i] = False
                num_uncovered += before_one
            else:
                before_zero = int(np.sum(cover_count[tids_i] == 0))
                cover_count[tids_i] += 1
                included[i] = True
                num_uncovered -= before_zero

            current_count = int(np.sum(included))
            if num_uncovered == 0 and current_count < best_count:
                best_num_uncovered = 0
                best_included = included.copy()
                best_count = current_count
            elif num_uncovered < best_num_uncovered:
                best_num_uncovered = num_uncovered
                best_included = included.copy()
                best_count = current_count

            if best_num_uncovered == 0 and best_count <= len(sol_indices):
                break

        return [i for i in range(num_k) if best_included[i]]

    # 7. Main search
    best_blocks = None
    best_len = float('inf')

    # Deterministic greedy
    sol = run_greedy(randomized=False)
    if sol is not None:
        sol = polish(sol)
        sol, _ = swap_search(sol)
        sol = polish(sol)
        sol, _ = pairwise_merge(sol)
        sol = polish(sol)
        best_blocks = sol[:]
        best_len = len(sol)

    # Randomized greedy restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 40.0:
            break

        sol = run_greedy(randomized=True, seed=i * 7 + 13)
        if sol is None:
            break

        sol = polish(sol)
        if len(sol) < best_len:
            sol, _ = swap_search(sol)
            sol = polish(sol)
            sol, _ = pairwise_merge(sol)
            sol = polish(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_blocks = sol[:]
        elif len(sol) <= best_len + 3:
            sol, _ = swap_search(sol)
            sol = polish(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_blocks = sol[:]

    # WalkSAT on best solution
    if best_blocks is not None and time.time() < deadline - 30.0:
        ws_sol = walksat(best_blocks, noise=0.25, max_steps=3000, time_limit=deadline - 25.0)
        ws_sol = polish(ws_sol)
        ws_sol, _ = swap_search(ws_sol)
        ws_sol = polish(ws_sol)
        ws_sol, _ = pairwise_merge(ws_sol)
        ws_sol = polish(ws_sol)
        if len(ws_sol) < best_len:
            best_len = len(ws_sol)
            best_blocks = ws_sol[:]

    # Second WalkSAT with different params
    if best_blocks is not None and time.time() < deadline - 20.0:
        ws_sol2 = walksat(best_blocks, noise=0.1, max_steps=2000, time_limit=deadline - 15.0)
        ws_sol2 = polish(ws_sol2)
        ws_sol2, _ = swap_search(ws_sol2)
        ws_sol2 = polish(ws_sol2)
        if len(ws_sol2) < best_len:
            best_len = len(ws_sol2)
            best_blocks = ws_sol2[:]

    # Final polish
    if best_blocks is not None:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            best_blocks = polish(best_blocks)
            best_blocks, _ = swap_search(best_blocks)
            best_blocks = polish(best_blocks)
            best_blocks, _ = pairwise_merge(best_blocks)
            best_blocks = polish(best_blocks)

    if best_blocks is None:
        return []

    return [list(all_kblocks[b]) for b in best_blocks]
