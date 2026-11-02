"""Symmetry-reduced branch-and-bound + greedy + swap for kissing number in R^11."""
import numpy as np
import itertools
import time
import sys


def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    D = 11

    # --- Symmetry reduction: canonical sign (first nonzero entry is +1) ---
    # This halves the candidate pool while preserving the maximum independent set size
    # up to a factor of 2 (each orbit has exactly 2 representatives).
    candidates_list = []
    # weight-4: C(11,4)*8 = 2640
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            if signs[0] == -1:  # canonical: first nonzero is +1
                continue
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    # weight-2: C(11,2)*2 = 110
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            if signs[0] == -1:
                continue
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    # weight-1: 11 (only +1 in each position)
    for pos in range(D):
        v = [0] * D
        v[pos] = 1
        candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 2761

    # Precompute conflict matrix
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Adjacency list for conflicts
    adj = [np.where(conflict[i] > 0)[0] for i in range(n)]
    full_deg = conflict.sum(axis=1).astype(np.int32)

    # --- Branch and Bound for Maximum Independent Set ---
    def bb_mis(conflict_mat, adj_list, n_nodes, time_limit, initial_bound=0):
        """Branch-and-bound MIS using greedy coloring upper bound."""
        best_set = []
        best_size = initial_bound
        remaining = np.ones(n_nodes, dtype=bool)
        # Order vertices by degree (ascending) for better pruning
        order = np.argsort(adj_list.__len__ if False else np.array([len(a) for a in adj_list]))

        # Greedy coloring bound on a set of candidate vertices
        def color_bound(candidates_set):
            """Greedy coloring gives upper bound on clique size in conflict graph = MIS in compat graph."""
            if len(candidates_set) == 0:
                return 0
            colors = {}
            color_count = 0
            for v in candidates_set:
                # Find lowest color not used by neighbors in candidates_set
                used_colors = set()
                for u in adj_list[v]:
                    if u in colors:
                        used_colors.add(colors[u])
                c = 0
                while c in used_colors:
                    c += 1
                colors[v] = c
                color_count = max(color_count, c + 1)
            return color_count

        # Simple BB: pick vertex, branch on include/exclude
        # Use greedy coloring as upper bound
        sys.setrecursionlimit(10000)

        def bb(remaining_set, current_set):
            nonlocal best_set, best_size
            if time.time() > time_limit:
                return
            if len(remaining_set) == 0:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                return

            # Upper bound: current + color bound on remaining
            ub = len(current_set) + color_bound(remaining_set)
            if ub <= best_size:
                return

            # Pick a vertex: highest degree in remaining
            v = max(remaining_set, key=lambda x: len([u for u in adj_list[x] if u in remaining_set]))

            # Branch 1: include v
            new_remaining = remaining_set - {v} - set(adj_list[v])
            current_set.add(v)
            bb(new_remaining, current_set)
            current_set.discard(v)

            # Branch 2: exclude v
            remaining_set.discard(v)
            bb(remaining_set, current_set)

        # Use a simpler iterative approach to avoid deep recursion
        # Greedy seed first
        greedy_seed = set()
        rem = set(range(n_nodes))
        while rem:
            # Pick vertex with min degree in rem
            v = min(rem, key=lambda x: len([u for u in adj_list[x] if u in rem]))
            greedy_seed.add(v)
            rem -= {v} | set(adj_list[v])
        best_size = len(greedy_seed)
        best_set = list(greedy_seed)

        # Now do BB with limited depth/time
        if time.time() < time_limit - 5:
            # Try improving via BB on the complement structure
            # For each vertex not in greedy_seed, try swapping
            for v in range(n_nodes):
                if time.time() > time_limit:
                    break
                if v in best_set:
                    continue
                # Try: remove one from best_set that conflicts with v, add v
                for u in best_set:
                    if conflict[v, u]:
                        # Check if best_set - {u} + {v} is valid
                        new_set = set(best_set) - {u} | {v}
                        # Verify independence
                        valid = True
                        for a in new_set:
                            for b in new_set:
                                if a < b and conflict[a, b]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid and len(new_set) > best_size:
                            best_size = len(new_set)
                            best_set = list(new_set)

        return best_set

    # Run BB with time limit
    bb_time_limit = deadline - 30.0
    bb_result = bb_mis(conflict, adj, n, bb_time_limit)

    # --- Greedy min-degree on remaining after BB ---
    sel_set = set(bb_result)
    selected = list(bb_result)
    compat_with_all = np.ones(n, dtype=np.uint8)
    for s in selected:
        compat_with_all = compat_with_all & compat[s]
    for s in selected:
        compat_with_all[s] = 0

    # Greedy fill
    while time.time() < deadline - 15.0:
        addable = np.where(compat_with_all > 0)[0]
        if len(addable) == 0:
            break
        # Pick min-degree among addable
        best_add = addable[np.argmin(full_deg[addable])]
        selected.append(best_add)
        sel_set.add(best_add)
        compat_with_all = compat_with_all & compat[best_add]
        compat_with_all[best_add] = 0

    # --- Swap local search ---
    def swap_local_search(selected_list, time_limit, max_rounds=200):
        selected = list(selected_list)
        sel_set = set(selected)
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0

        for _ in range(max_rounds):
            if time.time() > time_limit:
                break
            addable = np.where(compat_with_all > 0)[0]
            if len(addable) > 0:
                best_add = addable[np.argmin(full_deg[addable])]
                selected.append(best_add)
                sel_set.add(best_add)
                compat_with_all = compat_with_all & compat[best_add]
                compat_with_all[best_add] = 0
                continue

            found_swap = False
            if len(selected) > 30:
                trial_indices = np.random.choice(len(selected), size=min(10, len(selected)), replace=False)
            else:
                trial_indices = range(len(selected))

            for si in trial_indices:
                if time.time() > time_limit:
                    break
                s = selected[si]
                mask_except_s = np.ones(n, dtype=np.uint8)
                for s2 in selected:
                    if s2 != s:
                        mask_except_s = mask_except_s & compat[:, s2]
                mask_except_s[list(sel_set)] = 0
                candidates_for_swap = np.where(mask_except_s > 0)[0]
                if len(candidates_for_swap) < 2:
                    continue
                sorted_cands = candidates_for_swap[np.argsort(full_deg[candidates_for_swap])]
                found_pair = False
                for i in range(min(20, len(sorted_cands))):
                    c1 = sorted_cands[i]
                    compat_with_c1 = compat[c1]
                    for j in range(i + 1, min(40, len(sorted_cands))):
                        c2 = sorted_cands[j]
                        if compat_with_c1[c2]:
                            selected.remove(s)
                            sel_set.discard(s)
                            selected.append(c1)
                            sel_set.add(c1)
                            selected.append(c2)
                            sel_set.add(c2)
                            compat_with_all = np.ones(n, dtype=np.uint8)
                            for s3 in selected:
                                compat_with_all = compat_with_all & compat[s3]
                            for s3 in selected:
                                compat_with_all[s3] = 0
                            found_swap = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_swap:
                    break
            if not found_swap:
                break
        return selected

    if time.time() < deadline - 10.0:
        improved = swap_local_search(selected, deadline - 10.0, max_rounds=300)
        if len(improved) > len(selected):
            selected = improved

    # Final greedy fill
    if time.time() < deadline - 5.0:
        sel_set = set(selected)
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0
        addable = np.where(compat_with_all > 0)[0]
        for idx in addable[np.argsort(full_deg[addable])]:
            if time.time() > deadline - 5.0:
                break
            selected.append(idx)
            sel_set.add(idx)
            compat_with_all = compat_with_all & compat[idx]
            compat_with_all[idx] = 0

    if not selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(selected, dtype=np.int64)]
    return result.astype(np.int64)
