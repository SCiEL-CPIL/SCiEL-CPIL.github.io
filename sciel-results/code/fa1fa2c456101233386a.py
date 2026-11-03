"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 115.0  # hard wall-clock safety margin before the 150 s limit

def construct():
    t0 = time.perf_counter()
    best = None
    best_k = 0

    def now():
        return time.perf_counter()

    def elapsed():
        return now() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    def update_best(arr):
        nonlocal best, best_k
        if arr is None or arr.shape[0] == 0:
            return
        k = arr.shape[0]
        if k > best_k:
            best = arr.copy()
            best_k = k

    # ------------------------------------------------------------------
    # 1) Fast deterministic baseline: the full "2-weight" shell of norm^2=2
    #    220 vectors, all mutually valid (min dist = sqrt(2) = max norm)
    # ------------------------------------------------------------------
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base.append(v)
    base_arr = np.array(base, dtype=np.int64)
    update_best(base_arr)

    # ------------------------------------------------------------------
    # 2) Generate candidate vectors across multiple shells
    # ------------------------------------------------------------------
    def gen_shell(m):
        """Generate all integer vectors in R^D with L2^2 = m."""
        results = []
        if m == 2:
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        results.append(v)
        elif m == 3:
            for combo in itertools.combinations(range(D), 3):
                for signs in itertools.product([1, -1], repeat=3):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
        elif m == 4:
            for combo in itertools.combinations(range(D), 4):
                for signs in itertools.product([1, -1], repeat=4):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
            for i in range(D):
                for s in (2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s
                    results.append(v)
        elif m == 5:
            for combo in itertools.combinations(range(D), 5):
                for signs in itertools.product([1, -1], repeat=5):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
            for i in range(D):
                for j in range(i + 1, D):
                    for si in (1, -1):
                        for sj in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            results.append(v)
        return np.array(results, dtype=np.int64) if results else np.zeros((0, D), dtype=np.int64)

    # ------------------------------------------------------------------
    # 3) Core: vectorized conflict + minimum-degree greedy + local search
    # ------------------------------------------------------------------
    def solve_shell(V, max_ip_allowed):
        """Minimum-degree greedy then swap local search on conflict graph.
        V: (n, D) int array. max_ip_allowed: max inner product allowed between any pair.
        Returns best subset as (k, D) int array."""
        n = V.shape[0]
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)

        # Compute Gram matrix
        G = V.astype(np.int64) @ V.astype(np.int64).T
        # Conflict: ip > max_ip_allowed
        conflict = G > max_ip_allowed
        # Self-conflict: set diagonal to False
        np.fill_diagonal(conflict, False)

        # Degree of each node = number of conflicts
        degree = conflict.sum(axis=1).astype(np.int32)

        # --- Minimum-degree greedy ---
        selected = set()
        remaining = set(range(n))
        # Track which nodes are still in remaining for fast degree queries
        in_remaining = np.ones(n, dtype=bool)

        while remaining and not is_deadline():
            # Find node with minimum degree in remaining
            min_deg = min(degree[i] for i in remaining)
            # Among those with min degree, pick the first (deterministic)
            candidates = [i for i in remaining if degree[i] == min_deg]
            pick = min(candidates)  # deterministic tiebreak

            selected.add(pick)
            remaining.discard(pick)
            in_remaining[pick] = False

            # Remove all neighbors of pick from remaining
            neighbors = np.where(conflict[pick])[0]
            for nb in neighbors:
                if in_remaining[nb]:
                    remaining.discard(nb)
                    in_remaining[nb] = False

        greedy_result = sorted(selected)
        if not greedy_result:
            return np.zeros((0, D), dtype=np.int64)

        # --- Swap local search ---
        selected_set = set(greedy_result)
        selected_list = list(greedy_result)
        improved = True
        max_iters = 200
        iter_count = 0

        while improved and not is_deadline() and iter_count < max_iters:
            improved = False
            iter_count += 1
            current_size = len(selected_set)

            # For each selected node, try removing it and see if we can add 2+ more
            for i in range(len(selected_list)):
                if is_deadline():
                    break
                remove_node = selected_list[i]
                # Tentatively remove
                selected_set.remove(remove_node)

                # Find all nodes that can now be added (not in selected, not conflicting with selected)
                addable = []
                for j in range(n):
                    if j in selected_set or j == remove_node:
                        continue
                    # Check if j conflicts with any currently selected
                    can_add = True
                    for s in selected_set:
                        if conflict[j, s]:
                            can_add = False
                            break
                    if can_add:
                        addable.append(j)

                # From addable, greedily add as many as possible
                added = []
                remaining_add = set(addable)
                while remaining_add:
                    # Pick one with minimum conflicts among remaining_add
                    best_add = None
                    best_conf = 999999
                    for a in remaining_add:
                        conf_count = sum(1 for b in added if conflict[a, b])
                        if conf_count < best_conf:
                            best_conf = conf_count
                            best_add = a
                        if conf_count > 0:
                            continue
                    if best_add is None or best_conf > 0:
                        # Can't add any more without conflicts
                        break
                    added.append(best_add)
                    remaining_add.discard(best_add)
                    # Remove neighbors of best_add from remaining_add
                    nbs = np.where(conflict[best_add])[0]
                    for nb in nbs:
                        if nb in remaining_add:
                            remaining_add.discard(nb)

                # If we can add more than 1 (net gain > 0), do the swap
                if len(added) > 1:
                    for a in added:
                        selected_set.add(a)
                    selected_list = list(selected_set)
                    improved = True
                    break  # Restart the improvement loop

                # Undo the tentative removal
                selected_set.add(remove_node)

        result = V[sorted(selected_set)]
        return result

    # ------------------------------------------------------------------
    # 4) Vectorized mixed-shell solver
    # ------------------------------------------------------------------
    def solve_mixed(V, norms2):
        """Solve heterogeneous constraint: dist^2 >= max(norm_u^2, norm_v^2).
        V: (n, D) int array, norms2: (n,) int array of squared norms.
        """
        n = V.shape[0]
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)

        # Compute Gram matrix
        G = V.astype(np.int64) @ V.astype(np.int64).T

        # Conflict: 2*ip > min(norm_u^2, norm_v^2)
        # Build min_norm2 matrix: min(norms2[i], norms2[j]) for all pairs
        norms2_col = norms2[:, np.newaxis]
        norms2_row = norms2[np.newaxis, :]
        min_norm2 = np.minimum(norms2_col, norms2_row)

        # Conflict where 2*G > min_norm2
        conflict = 2 * G > min_norm2
        np.fill_diagonal(conflict, False)

        # Degree of each node
        degree = conflict.sum(axis=1).astype(np.int32)

        # --- Minimum-degree greedy ---
        selected = set()
        remaining = set(range(n))
        in_remaining = np.ones(n, dtype=bool)

        while remaining and not is_deadline():
            # Find node with minimum degree in remaining
            min_deg = min(degree[i] for i in remaining)
            candidates = [i for i in remaining if degree[i] == min_deg]
            pick = min(candidates)

            selected.add(pick)
            remaining.discard(pick)
            in_remaining[pick] = False

            # Remove all neighbors of pick from remaining
            neighbors = np.where(conflict[pick])[0]
            for nb in neighbors:
                if in_remaining[nb]:
                    remaining.discard(nb)
                    in_remaining[nb] = False

        greedy_result = sorted(selected)
        if not greedy_result:
            return np.zeros((0, D), dtype=np.int64)

        # --- Swap local search ---
        selected_set = set(greedy_result)
        selected_list = list(greedy_result)
        improved = True
        max_iters = 100
        iter_count = 0

        while improved and not is_deadline() and iter_count < max_iters:
            improved = False
            iter_count += 1

            for i in range(len(selected_list)):
                if is_deadline():
                    break
                remove_node = selected_list[i]
                selected_set.remove(remove_node)

                # Find all nodes that can now be added
                addable = []
                for j in range(n):
                    if j in selected_set or j == remove_node:
                        continue
                    can_add = True
                    for s in selected_set:
                        if conflict[j, s]:
                            can_add = False
                            break
                    if can_add:
                        addable.append(j)

                # Greedily add as many as possible
                added = []
                remaining_add = set(addable)
                while remaining_add:
                    best_add = None
                    best_conf = 999999
                    for a in remaining_add:
                        conf_count = sum(1 for b in added if conflict[a, b])
                        if conf_count < best_conf:
                            best_conf = conf_count
                            best_add = a
                        if conf_count > 0:
                            continue
                    if best_add is None or best_conf > 0:
                        break
                    added.append(best_add)
                    remaining_add.discard(best_add)
                    nbs = np.where(conflict[best_add])[0]
                    for nb in nbs:
                        if nb in remaining_add:
                            remaining_add.discard(nb)

                if len(added) > 1:
                    for a in added:
                        selected_set.add(a)
                    selected_list = list(selected_set)
                    improved = True
                    break

                selected_set.add(remove_node)

        result = V[sorted(selected_set)]
        return result

    # ------------------------------------------------------------------
    # 5) Try each shell independently
    # ------------------------------------------------------------------
    # Shell m=2: 220 vectors, max_ip_allowed = 1
    V2 = gen_shell(2)
    try:
        res2 = solve_shell(V2, 1)
        update_best(res2)
    except Exception:
        pass

    # Shell m=3: 1320 vectors, max_ip_allowed = 1
    V3 = gen_shell(3)
    try:
        res3 = solve_shell(V3, 1)
        update_best(res3)
    except Exception:
        pass

    # Shell m=4: 5302 vectors, max_ip_allowed = 2
    if not is_deadline():
        V4 = gen_shell(4)
        if V4.shape[0] <= 6000:
            try:
                res4 = solve_shell(V4, 2)
                update_best(res4)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 6) Mixed-shell approach: combine m=2 and m=3
    # ------------------------------------------------------------------
    if not is_deadline():
        combined = np.vstack([V2, V3]) if V3.shape[0] > 0 else V2
        n_combined = combined.shape[0]
        norms2_combined = np.sum(combined.astype(np.int64)**2, axis=1)

        if n_combined <= 2000:
            try:
                res_mixed = solve_mixed(combined, norms2_combined)
                update_best(res_mixed)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 7) Mixed-shell approach: combine m=2, m=3, and m=4
    # ------------------------------------------------------------------
    if not is_deadline():
        V4 = gen_shell(4)
        if V4.shape[0] > 0:
            combined2 = np.vstack([V2, V3, V4])
            n_combined2 = combined2.shape[0]
            norms2_combined2 = np.sum(combined2.astype(np.int64)**2, axis=1)

            if n_combined2 <= 8000:
                try:
                    res_mixed2 = solve_mixed(combined2, norms2_combined2)
                    update_best(res_mixed2)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 8) Return best
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best
