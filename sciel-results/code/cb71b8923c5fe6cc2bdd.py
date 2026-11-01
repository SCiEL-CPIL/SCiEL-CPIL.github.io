import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute block_tsubs and tsub_blocks
    block_tsubs = []
    for B in all_kblocks:
        covered = []
        for ts in itertools.combinations(B, t):
            covered.append(tsub_to_idx[ts])
        block_tsubs.append(covered)

    tsub_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, tsubs in enumerate(block_tsubs):
        for s_idx in tsubs:
            tsub_blocks[s_idx].append(b_idx)

    # Convert tsub_blocks to numpy arrays for speed
    tsub_blocks_arr = [np.array(tb, dtype=np.int32) for tb in tsub_blocks]
    # block_tsubs as numpy arrays
    block_tsubs_arr = [np.array(bt, dtype=np.int32) for bt in block_tsubs]

    best_solution = None
    best_score = float('inf')

    num_restarts = 8

    for restart in range(num_restarts):
        if time.time() > deadline - 25.0:
            break

        # Greedy
        uncovered = np.ones(n_tsubs, dtype=bool)
        gains = np.full(n_blocks, len(block_tsubs[0]), dtype=np.int32)
        selected_indices = []
        while np.any(uncovered):
            if time.time() > deadline - 20.0:
                break
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            b_idx = np.argmax(gains)
            selected_indices.append(b_idx)
            for s_idx in block_tsubs_arr[b_idx]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    for b2 in tsub_blocks_arr[s_idx]:
                        gains[b2] -= 1
        if np.any(uncovered):
            continue

        # Local search
        selected_set = set(selected_indices)
        selected_list = list(selected_indices)
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for b in selected_list:
            cover_counts[block_tsubs_arr[b]] += 1

        improved = True
        local_time_limit = deadline - 8.0
        iteration = 0
        while improved and time.time() < local_time_limit:
            iteration += 1
            improved = False

            # 1. Remove redundant blocks
            new_selected = []
            for b in selected_list:
                essential = False
                for s in block_tsubs_arr[b]:
                    if cover_counts[s] == 1:
                        essential = True
                        break
                if essential:
                    new_selected.append(b)
                else:
                    cover_counts[block_tsubs_arr[b]] -= 1
                    selected_set.remove(b)
                    improved = True
            selected_list = new_selected

            if not improved:
                # 2. Compute unique t-subsets per block
                unique_map = {}
                for b in selected_list:
                    u_list = [s for s in block_tsubs_arr[b] if cover_counts[s] == 1]
                    if u_list:
                        unique_map[b] = u_list

                if not unique_map:
                    break

                # 3. Try 1-for-1 swaps
                candidates = sorted(unique_map.keys(), key=lambda x: len(unique_map[x]))
                swap_made = False
                for b_out in candidates[:8]:
                    if time.time() > local_time_limit:
                        break
                    u_list = unique_map[b_out]
                    cands_in = tsub_blocks_arr[u_list[0]]
                    found = None
                    for b_in in cands_in:
                        if b_in in selected_set:
                            continue
                        ok = True
                        for s in u_list[1:]:
                            if s not in block_tsubs_arr[b_in]:
                                ok = False
                                break
                        if ok:
                            found = b_in
                            break
                    if found is not None:
                        selected_list.remove(b_out)
                        selected_set.remove(b_out)
                        cover_counts[block_tsubs_arr[b_out]] -= 1
                        selected_list.append(found)
                        selected_set.add(found)
                        cover_counts[block_tsubs_arr[found]] += 1
                        improved = True
                        swap_made = True
                        break

                if swap_made:
                    continue

                # 4. Try 2-for-1 merges (reduce block count by 1)
                merge_made = False
                cand_list = sorted(unique_map.keys(), key=lambda x: len(unique_map[x]))
                for i in range(min(len(cand_list), 12)):
                    if time.time() > local_time_limit:
                        break
                    b1 = cand_list[i]
                    u1 = unique_map[b1]
                    for j in range(i+1, min(len(cand_list), 12)):
                        b2 = cand_list[j]
                        u2 = unique_map[b2]
                        all_u = list(set(u1) | set(u2))
                        # Find a block that covers all_u and is not selected
                        cands = tsub_blocks_arr[all_u[0]]
                        found = None
                        for b_new in cands:
                            if b_new in selected_set or b_new == b1 or b_new == b2:
                                continue
                            ok = True
                            for s in all_u[1:]:
                                if s not in block_tsubs_arr[b_new]:
                                    ok = False
                                    break
                            if ok:
                                found = b_new
                                break
                        if found is not None:
                            # Verify: after removing b1, b2, adding found, all covered
                            # Quick check: for each t-subset, is count still >=1?
                            # Temporarily adjust
                            test_counts = cover_counts.copy()
                            test_counts[block_tsubs_arr[b1]] -= 1
                            test_counts[block_tsubs_arr[b2]] -= 1
                            test_counts[block_tsubs_arr[found]] += 1
                            if np.min(test_counts) >= 0:
                                # Apply
                                selected_list.remove(b1)
                                selected_list.remove(b2)
                                selected_set.remove(b1)
                                selected_set.remove(b2)
                                cover_counts[block_tsubs_arr[b1]] -= 1
                                cover_counts[block_tsubs_arr[b2]] -= 1
                                selected_list.append(found)
                                selected_set.add(found)
                                cover_counts[block_tsubs_arr[found]] += 1
                                improved = True
                                merge_made = True
                                break
                    if merge_made:
                        break

                if merge_made:
                    continue

                # 5. Try 2-for-2 swaps (structural change)
                swap2_made = False
                for i in range(min(len(cand_list), 8)):
                    if time.time() > local_time_limit:
                        break
                    b1 = cand_list[i]
                    u1 = unique_map[b1]
                    for j in range(i+1, min(len(cand_list), 8)):
                        b2 = cand_list[j]
                        u2 = unique_map[b2]
                        all_u = list(set(u1) | set(u2))
                        # Find two blocks not in selected that cover all_u
                        cands = tsub_blocks_arr[all_u[0]]
                        found_blocks = []
                        for b_new in cands:
                            if b_new in selected_set or b_new == b1 or b_new == b2:
                                continue
                            ok = True
                            for s in all_u[1:]:
                                if s not in block_tsubs_arr[b_new]:
                                    ok = False
                                    break
                            if ok:
                                found_blocks.append(b_new)
                                if len(found_blocks) == 2:
                                    break
                        if len(found_blocks) == 2:
                            b_new1, b_new2 = found_blocks
                            # Check: after removing b1,b2 and adding b_new1,b_new2, all covered
                            test_counts = cover_counts.copy()
                            test_counts[block_tsubs_arr[b1]] -= 1
                            test_counts[block_tsubs_arr[b2]] -= 1
                            test_counts[block_tsubs_arr[b_new1]] += 1
                            test_counts[block_tsubs_arr[b_new2]] += 1
                            if np.min(test_counts) >= 0:
                                selected_list.remove(b1)
                                selected_list.remove(b2)
                                selected_set.remove(b1)
                                selected_set.remove(b2)
                                cover_counts[block_tsubs_arr[b1]] -= 1
                                cover_counts[block_tsubs_arr[b2]] -= 1
                                selected_list.append(b_new1)
                                selected_list.append(b_new2)
                                selected_set.add(b_new1)
                                selected_set.add(b_new2)
                                cover_counts[block_tsubs_arr[b_new1]] += 1
                                cover_counts[block_tsubs_arr[b_new2]] += 1
                                improved = True
                                swap2_made = True
                                break
                    if swap2_made:
                        break

                if swap2_made:
                    continue

                # No improvement possible
                break

        current_size = len(selected_list)
        if current_size < best_score:
            best_score = current_size
            best_solution = selected_list[:]

        if best_score <= 131:
            break

    if best_solution is None:
        # Fallback greedy
        uncovered = np.ones(n_tsubs, dtype=bool)
        gains = np.full(n_blocks, len(block_tsubs[0]), dtype=np.int32)
        selected_indices = []
        while np.any(uncovered):
            b_idx = np.argmax(gains)
            if gains[b_idx] == 0:
                break
            selected_indices.append(b_idx)
            for s_idx in block_tsubs_arr[b_idx]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    for b2 in tsub_blocks_arr[s_idx]:
                        gains[b2] -= 1
        best_solution = selected_indices

    blocks = [all_kblocks[i] for i in best_solution]
    return blocks
