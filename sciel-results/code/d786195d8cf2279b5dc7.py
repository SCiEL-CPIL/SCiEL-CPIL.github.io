import time
import random
import math
from collections import Counter
import heapq

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 140.0
    row_len = k + 1
    num_diffs_per_row = k * (k + 1) // 2  # C(k+1, 2)
    total_diffs_needed = n_rows * num_diffs_per_row

    def row_mask(row):
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= (1 << (row[j] - ri))
        return m

    def greedy_row(used_mask):
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            new_mask = 0
            for a in row:
                new_mask |= (1 << (x - a))
            if (new_mask & used_mask) == 0:
                row.append(x)
                used_mask |= new_mask
        return row

    def find_row_dfs(used_mask, best_max_init, S_cap, node_budget, dl):
        best = [None]
        best_max = [best_max_init]
        nodes = [0]

        def dfs(row, rmask, last):
            if time.time() > dl or nodes[0] >= node_budget:
                return
            nodes[0] += 1
            if len(row) == row_len:
                m = row[-1]
                if m < best_max[0]:
                    best_max[0] = m
                    best[0] = row[:]
                return
            hi = min(best_max[0] - 1, S_cap)
            lo = last + 1
            if lo > hi:
                return
            for x in range(lo, hi + 1):
                new_mask = 0
                for a in row:
                    new_mask |= (1 << (x - a))
                if (new_mask & used_mask) or (new_mask & rmask):
                    continue
                row.append(x)
                dfs(row, rmask | new_mask, x)
                row.pop()
                if nodes[0] >= node_budget or time.time() > dl:
                    return

        dfs([0], 0, 0)
        if best[0] is None:
            return None
        return best[0], row_mask(best[0])

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def build_diff_counts(rows):
        counts = Counter()
        for r in rows:
            for i in range(len(r)):
                ri = r[i]
                for j in range(i + 1, len(r)):
                    d = r[j] - ri
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def remove_row_diffs(dc, row):
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ri
                dc[d] -= 1
                if dc[d] == 0:
                    del dc[d]

    def add_row_diffs(dc, row):
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ri
                dc[d] = dc.get(d, 0) + 1

    def row_rebuild_greedy(used_mask, max_scope, rng, attempts=3):
        for attempt in range(attempts):
            row = [0]
            x = 0
            local_mask = 0
            for j in range(1, row_len):
                found = False
                start = x + 1
                if attempt > 0 and rng.random() < 0.3:
                    start = x + rng.randint(1, 3)
                for t in range(start, max_scope + 1):
                    new_mask = 0
                    for a in row:
                        new_mask |= (1 << (t - a))
                    if (new_mask & used_mask) or (new_mask & local_mask):
                        continue
                    row.append(t)
                    local_mask |= new_mask
                    x = t
                    found = True
                    break
                if not found:
                    return None
            return row, row_mask(row)
        return None

    def sa_optimize(rows, k, iters, rng, deadline):
        n = len(rows)
        cur_rows = [list(r) for r in rows]
        cur_scope = scope_of(cur_rows)
        best_rows = [list(r) for r in cur_rows]
        best_scope = cur_scope

        diff_counts = build_diff_counts(cur_rows)
        row_masks = [row_mask(r) for r in cur_rows]
        xor_all = 0
        for m in row_masks:
            xor_all ^= m

        def compute_used_without(i):
            return xor_all ^ row_masks[i]

        def get_cost(rows, dc):
            scope = scope_of(rows)
            collisions = sum(max(0, c - 1) for c in dc.values())
            return scope + 1000 * collisions

        cur_cost = get_cost(cur_rows, diff_counts)
        best_cost = cur_cost

        T = 5.0
        cooling = 0.9997
        min_T = 0.05

        for it in range(iters):
            if time.time() > deadline:
                break

            r = rng.random()

            if r < 0.25:
                i = rng.randint(0, n - 1)
                old_row = list(cur_rows[i])
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                res = row_rebuild_greedy(used_without, cur_scope + 3, rng)
                if res is None:
                    continue
                new_row, new_mask = res

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                row_masks[i] = new_mask
                xor_all ^= old_mask ^ new_mask
                new_cost = get_cost(cur_rows, diff_counts)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = scope_of(cur_rows)
                        best_rows = [list(r) for r in cur_rows]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row
                    row_masks[i] = old_mask
                    xor_all ^= new_mask ^ old_mask

            elif r < 0.50:
                max_val = cur_scope
                max_row_idx = -1
                for ri in range(n):
                    if cur_rows[ri][-1] == max_val:
                        max_row_idx = ri
                        break
                if max_row_idx < 0:
                    continue

                i = max_row_idx
                old_row = list(cur_rows[i])
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                res = find_row_dfs(used_without, old_row[-1], old_row[-1] - 1, 50000, min(deadline, time.time() + 2))
                if res is not None:
                    new_row, new_mask = res
                    remove_row_diffs(diff_counts, old_row)
                    add_row_diffs(diff_counts, new_row)
                    cur_rows[i] = new_row
                    row_masks[i] = new_mask
                    xor_all ^= old_mask ^ new_mask
                    new_cost = get_cost(cur_rows, diff_counts)

                    if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                        cur_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_scope = scope_of(cur_rows)
                            best_rows = [list(r) for r in cur_rows]
                    else:
                        remove_row_diffs(diff_counts, new_row)
                        add_row_diffs(diff_counts, old_row)
                        cur_rows[i] = old_row
                        row_masks[i] = old_mask
                        xor_all ^= new_mask ^ old_mask

            else:
                i = rng.randint(0, n - 1)
                j = rng.randint(1, row_len - 1)
                old_row = list(cur_rows[i])
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                lo = old_row[j - 1] + 1
                hi = old_row[j + 1] - 1 if j < row_len - 1 else cur_scope + 2
                if lo >= hi:
                    continue
                new_val = rng.randint(lo, hi)
                if new_val == old_row[j]:
                    continue

                new_row = list(old_row)
                new_row[j] = new_val
                if new_row[j] <= new_row[j-1] or (j < row_len-1 and new_row[j] >= new_row[j+1]):
                    continue

                new_mask = row_mask(new_row)
                if new_mask & used_without:
                    continue

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                row_masks[i] = new_mask
                xor_all ^= old_mask ^ new_mask
                new_cost = get_cost(cur_rows, diff_counts)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = scope_of(cur_rows)
                        best_rows = [list(r) for r in cur_rows]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row
                    row_masks[i] = old_mask
                    xor_all ^= new_mask ^ old_mask

            T = max(min_T, T * cooling)

        return best_scope, best_rows

    def a_star_search(used_mask, rows_remaining, max_scope, node_budget, dl):
        """A* search over bitmask states to find minimal scope for remaining rows."""
        if rows_remaining == 0:
            return [], 0

        # Heuristic: minimum scope needed given remaining differences
        # Each row needs num_diffs_per_row distinct differences from [1, scope]
        # With scope S, we have S available difference values
        # Total remaining diffs = rows_remaining * num_diffs_per_row
        # Already used diffs in [1, S]: popcount of used_mask & ((1 << (S+1)) - 1)
        # Remaining available = S - used_in_range
        # Need: rows_remaining * num_diffs_per_row <= remaining_available
        # => S >= used_in_range + rows_remaining * num_diffs_per_row

        def heuristic(um, rem_rows, scope):
            # Count used diffs in [1, scope]
            mask_range = (1 << (scope + 1)) - 1
            used_in_range = bin(um & mask_range).count('1')
            needed = rem_rows * num_diffs_per_row
            # Minimum scope such that scope - used_in_range >= needed
            # scope >= used_in_range + needed
            return used_in_range + needed

        # State: (f, g, used_mask, rows_built, scope)
        # g = current scope (cost so far)
        # f = g + h
        start_scope = heuristic(used_mask, rows_remaining, 0)
        # We search for minimal scope S such that we can build rows_remaining rows
        # within scope S without reusing differences

        best_result = [None]
        nodes_expanded = [0]

        # Priority queue: (f, g, used_mask, rows_built_list, scope)
        # We'll do iterative deepening on scope
        # For each candidate scope S, try to see if we can build all remaining rows

        def try_scope(S, um, rem_rows, rows_built, budget, dl_check):
            """Try to build rem_rows rows within scope S using DFS with pruning."""
            if rem_rows == 0:
                return True
            if time.time() > dl_check or budget[0] <= 0:
                return False

            # Find one valid row within scope S
            def find_one_row(um, S, budget, dl_check):
                if time.time() > dl_check or budget[0] <= 0:
                    return None
                budget[0] -= 1

                def dfs_row(row, rmask, last):
                    if time.time() > dl_check or budget[0] <= 0:
                        return None
                    budget[0] -= 1
                    if len(row) == row_len:
                        return row[:]
                    hi = min(S, last + 10)  # Limit branching
                    lo = last + 1
                    if lo > hi:
                        return None
                    for x in range(lo, hi + 1):
                        new_mask = 0
                        for a in row:
                            new_mask |= (1 << (x - a))
                        if (new_mask & um) or (new_mask & rmask):
                            continue
                        row.append(x)
                        res = dfs_row(row, rmask | new_mask, x)
                        row.pop()
                        if res is not None:
                            return res
                    return None

                return dfs_row([0], 0, 0)

            # Greedy approach: repeatedly find a row
            local_rows = []
            local_um = um
            for _ in range(rem_rows):
                if time.time() > dl_check or budget[0] <= 0:
                    return False
                row = find_one_row(local_um, S, budget, dl_check)
                if row is None:
                    return False
                local_rows.append(row)
                local_um |= row_mask(row)
            return True

        # Iterative deepening on scope
        lower_bound = max(heuristic(used_mask, rows_remaining, 0), 1)
        upper_bound = max_scope

        # Binary search or linear search on scope
        best_S = None
        lo_S = lower_bound
        hi_S = upper_bound

        # Try to find minimal S
        while lo_S <= hi_S and time.time() < dl:
            mid_S = (lo_S + hi_S) // 2
            budget = [node_budget // 4]
            if try_scope(mid_S, used_mask, rows_remaining, [], budget, dl):
                best_S = mid_S
                hi_S = mid_S - 1
            else:
                lo_S = mid_S + 1

        if best_S is not None:
            # Reconstruct: build rows within best_S
            budget = [node_budget // 2]
            local_rows = []
            local_um = used_mask
            for _ in range(rows_remaining):
                if time.time() > dl or budget[0] <= 0:
                    break
                row = None
                # Use DFS to find a row
                def dfs_row_full(row, rmask, last, um, S, budget, dl):
                    if time.time() > dl or budget[0] <= 0:
                        return None
                    budget[0] -= 1
                    if len(row) == row_len:
                        return row[:]
                    hi = S
                    lo = last + 1
                    if lo > hi:
                        return None
                    for x in range(lo, hi + 1):
                        new_mask = 0
                        for a in row:
                            new_mask |= (1 << (x - a))
                        if (new_mask & um) or (new_mask & rmask):
                            continue
                        row.append(x)
                        res = dfs_row_full(row, rmask | new_mask, x, um, S, budget, dl)
                        row.pop()
                        if res is not None:
                            return res
                    return None

                row = dfs_row_full([0], 0, 0, local_um, best_S, budget, dl)
                if row is None:
                    break
                local_rows.append(row)
                local_um |= row_mask(row)

            if len(local_rows) == rows_remaining:
                return local_rows, best_S

        return None, None

    def run(seed, dl):
        rng = random.Random(seed)
        used_mask = 0
        rows = []

        for _ in range(n_rows):
            if time.time() > dl:
                break
            res = find_row_dfs(used_mask, 200, 200, 100000, dl)
            if res is None:
                row = greedy_row(used_mask)
                used_mask |= row_mask(row)
            else:
                row, rm = res
                used_mask |= rm
            rows.append(row)
        while len(rows) < n_rows:
            row = greedy_row(used_mask)
            used_mask |= row_mask(row)
            rows.append(row)

        best_rows = [r[:] for r in rows]
        best_scope = scope_of(rows)

        if time.time() < dl - 5:
            sa_scope, sa_rows = sa_optimize(rows, k, 5000, rng, dl - 2)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [r[:] for r in sa_rows]

        # A* / iterative deepening phase: try to reduce scope
        if time.time() < dl - 3:
            cur = [list(r) for r in best_rows]
            n = len(cur)
            cur_masks = [row_mask(r) for r in cur]
            improved = True
            while improved and time.time() < dl - 2:
                improved = False
                for i in range(n):
                    if time.time() >= dl - 2:
                        break
                    cur_scope_i = cur[i][-1]
                    if cur_scope_i <= 10:
                        continue
                    used_wo = 0
                    for ri in range(n):
                        if ri != i:
                            used_wo |= cur_masks[ri]
                    # Try to find a smaller row using A*-guided search
                    target_scope = cur_scope_i - 1
                    budget = [80000]
                    res_a, _ = a_star_search(used_wo, 1, target_scope, 80000, min(dl - 2, time.time() + 2))
                    if res_a is not None and len(res_a) == 1:
                        new_row = res_a[0]
                        new_mask = row_mask(new_row)
                        if new_row[-1] < cur[i][-1]:
                            cur[i] = new_row
                            cur_masks[i] = new_mask
                            improved = True
            if scope_of(cur) < best_scope:
                best_scope = scope_of(cur)
                best_rows = [r[:] for r in cur]

        return best_scope, best_rows

    try:
        overall = None
        seed = 0
        first = True
        while time.time() < DEADLINE - 5:
            if first:
                dl = min(DEADLINE - 3, time.time() + 45)
                first = False
            else:
                dl = min(DEADLINE - 3, time.time() + 10)
            sc, rows = run(seed, dl)
            if overall is None or sc < overall[0]:
                overall = (sc, rows)
            seed += 1
        if overall is not None:
            return overall[1]
    except Exception:
        pass

    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row = greedy_row(used_mask)
        used_mask |= row_mask(row)
        rows.append(row)
    return rows
