import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))

    # 1. Precomputation
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)

    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)

    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)

    def run_greedy(seed):
        random.seed(seed)
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []

        # Use a max-heap: entries are (-gain, block_idx)
        # We'll use lazy deletion: push new entries, skip stale ones on pop
        heap = [(-int(g), i) for i, g in enumerate(block_gain)]
        heapq.heapify(heap)

        while uncovered_count > 0 and time.time() < DEADLINE:
            # Pop until we find a valid (non-stale) entry
            while heap:
                neg_gain, bidx = heapq.heappop(heap)
                gain = -neg_gain
                if gain == block_gain[bidx] and gain > 0:
                    break
            else:
                break  # heap exhausted

            if gain <= 0:
                break

            selected_indices.append(bidx)

            # Update coverage and gains
            subs = block_tsubs[bidx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    # Update gains for all blocks covering this t-subset
                    for nb in tsub_blocks[si]:
                        block_gain[nb] -= 1
                        # Push updated gain to heap (lazy deletion)
                        if block_gain[nb] > 0:
                            heapq.heappush(heap, (-int(block_gain[nb]), nb))

        return selected_indices

    def reduce_solution_fast(selected_indices):
        if not selected_indices:
            return []

        n_sel = len(selected_indices)

        # Build global_to_local as a numpy array for O(1) access
        # global_to_local_arr[g] = local index, or -1 if not selected
        global_to_local_arr = np.full(N_blocks, -1, dtype=np.int32)
        for i, g in enumerate(selected_indices):
            global_to_local_arr[g] = i

        active = np.ones(n_sel, dtype=np.bool_)

        total_count = np.zeros(N_t, dtype=np.int32)
        crit_count = np.zeros(n_sel, dtype=np.int32)

        # Compute total_count
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                total_count[si] += 1

        # Compute crit_count
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                if total_count[si] == 1:
                    crit_count[i] += 1

        # Queue of removable blocks (crit_count == 0)
        queue = [i for i in range(n_sel) if crit_count[i] == 0]

        while queue:
            i = queue.pop()
            if not active[i] or crit_count[i] > 0:
                continue

            # Remove block i
            active[i] = False
            g = selected_indices[i]

            for si in block_tsubs[g]:
                old_c = total_count[si]
                total_count[si] = old_c - 1
                new_c = old_c - 1

                if old_c == 2 and new_c == 1:
                    # si is now critical. Update crit_count for all other active blocks containing si.
                    for g_other in tsub_blocks[si]:
                        if g_other == g:
                            continue
                        i_other = global_to_local_arr[g_other]
                        if i_other >= 0 and active[i_other]:
                            crit_count[i_other] += 1

        return [selected_indices[i] for i in range(n_sel) if active[i]]

    def local_search_swap(selected_indices, max_iters=200):
        """
        Try to improve by removing a block and adding a different one that covers
        all t-subsets that were uniquely covered by the removed block.
        """
        if not selected_indices or len(selected_indices) < 2:
            return selected_indices

        n_sel = len(selected_indices)
        global_to_local_arr = np.full(N_blocks, -1, dtype=np.int32)
        for i, g in enumerate(selected_indices):
            global_to_local_arr[g] = i

        total_count = np.zeros(N_t, dtype=np.int32)
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                total_count[si] += 1

        # For each selected block, find its "unique" t-subsets (count == 1)
        # and the set of candidate replacement blocks that cover all of them.
        # We do a simple pass: try removing each block, see if we can add one back.

        best_list = list(selected_indices)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > DEADLINE - 5.0:
                break

            n_sel = len(best_list)
            # Rebuild structures
            global_to_local_arr = np.full(N_blocks, -1, dtype=np.int32)
            for i, g in enumerate(best_list):
                global_to_local_arr[g] = i

            total_count = np.zeros(N_t, dtype=np.int32)
            for i in range(n_sel):
                g = best_list[i]
                for si in block_tsubs[g]:
                    total_count[si] += 1

            # Try removing each block
            for i in range(n_sel):
                if time.time() > DEADLINE - 5.0:
                    break
                g_remove = best_list[i]
                unique_subs = []
                for si in block_tsubs[g_remove]:
                    if total_count[si] == 1:
                        unique_subs.append(si)

                if not unique_subs:
                    # Can remove freely
                    new_list = best_list[:i] + best_list[i+1:]
                    best_list = new_list
                    improved = True
                    break

                # Need to find a block that covers all unique_subs
                # and is not already in best_list
                # Candidate: must contain all points in each unique sub
                # Use the first unique sub to filter candidates
                first_sub_set = set(all_t[unique_subs[0]])
                found = False
                for si in unique_subs[1:]:
                    other_set = set(all_t[si])
                    first_sub_set &= other_set
                    if not first_sub_set:
                        break

                if first_sub_set:
                    # Find blocks containing all points in first_sub_set
                    # This is expensive; skip for now unless small
                    pass

        return best_list

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol = [all_blocks[i] for i in sol_idx]
    best_len = len(best_sol)

    # Randomized restarts
    while time.time() < DEADLINE - 5.0:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy(seed)
        cand_idx = reduce_solution_fast(cand_idx)
        if len(cand_idx) < best_len:
            best_len = len(cand_idx)
            best_sol = [all_blocks[i] for i in cand_idx]

    return best_sol
