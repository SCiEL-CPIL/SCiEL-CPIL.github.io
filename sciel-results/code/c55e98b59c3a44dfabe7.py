import time
import random
import math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 145.0
    row_len = k + 1
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    def row_mask(row):
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= (1 << (row[j] - ri))
        return m

    def row_diff_list(row):
        diffs = []
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                diffs.append(row[j] - ri)
        return diffs

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != row_len:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def greedy_row(used_mask, rng=None, randomize=False, max_val=500):
        row = [0]
        x = 0
        local_mask = 0
        while len(row) < row_len:
            found = False
            if randomize and rng and len(row) > 1:
                skip = rng.randint(0, 2)
                start = x + 1 + skip
            else:
                start = x + 1
            for t in range(start, max_val):
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

    def build_diff_counts(rows):
        counts = {}
        for r in rows:
            for d in row_diff_list(r):
                counts[d] = counts.get(d, 0) + 1
        return counts

    def remove_row_diffs(dc, row):
        for d in row_diff_list(row):
            dc[d] -= 1
            if dc[d] == 0:
                del dc[d]

    def add_row_diffs(dc, row):
        for d in row_diff_list(row):
            dc[d] = dc.get(d, 0) + 1

    def get_cost(rows, dc):
        scope = scope_of(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        return scope + 1000 * collisions

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

    def try_swap_two_entries(cur_rows, diff_counts, rng):
        i1 = rng.randint(0, n_rows - 1)
        j1 = rng.randint(1, row_len - 1)
        i2 = rng.randint(0, n_rows - 1)
        j2 = rng.randint(1, row_len - 1)
        
        if i1 == i2 and j1 == j2:
            return False
        
        row1 = cur_rows[i1]
        row2 = cur_rows[i2]
        v1 = row1[j1]
        v2 = row2[j2]
        
        if v1 == v2:
            return False
        
        new_row1 = list(row1)
        new_row2 = list(row2)
        new_row1[j1] = v2
        new_row2[j2] = v1
        
        for idx in range(len(new_row1) - 1):
            if new_row1[idx] >= new_row1[idx + 1]:
                return False
        for idx in range(len(new_row2) - 1):
            if new_row2[idx] >= new_row2[idx + 1]:
                return False
        
        mask1_new = row_mask(new_row1)
        mask2_new = row_mask(new_row2)
        
        used_without = 0
        for ri in range(n_rows):
            if ri != i1 and ri != i2:
                used_without |= row_mask(cur_rows[ri])
        
        if (mask1_new & used_without) or (mask2_new & used_without):
            return False
        
        if (mask1_new & mask2_new):
            return False
        
        remove_row_diffs(diff_counts, row1)
        remove_row_diffs(diff_counts, row2)
        add_row_diffs(diff_counts, new_row1)
        add_row_diffs(diff_counts, new_row2)
        
        cur_rows[i1] = new_row1
        cur_rows[i2] = new_row2
        
        return True

    # Phase 1: Greedy construction with multiple seeds
    best_rows = None
    best_scope = float('inf')
    rng = random.Random(42)
    
    for seed in range(40):
        if time.time() >= DEADLINE - 25:
            break
        r = random.Random(seed * 1000 + 42)
        used_mask = 0
        rows = []
        success = True
        for _ in range(n_rows):
            res = greedy_row(used_mask, r, randomize=(seed > 3), max_val=400)
            if res is None:
                success = False
                break
            row, mask = res
            rows.append(row)
            used_mask |= mask
        if success:
            scope = scope_of(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(rr) for rr in rows]

    if best_rows is None:
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(1, k + 1):
                v = row[-1] + 1
                while True:
                    valid = True
                    for prev in row:
                        d = v - prev
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(v)
                        for prev in row[:-1]:
                            used_diffs.add(v - prev)
                        break
                    v += 1
            rows.append(row)
        return rows

    # Phase 2: Simulated Annealing
    if time.time() < DEADLINE - 15:
        cur_rows = [list(r) for r in best_rows]
        cur_scope = scope_of(cur_rows)
        best_rows_sa = [list(r) for r in cur_rows]
        best_scope_sa = cur_scope
        
        diff_counts = build_diff_counts(cur_rows)
        cur_cost = get_cost(cur_rows, diff_counts)
        best_cost = cur_cost

        T = 8.0
        cooling = 0.9998
        min_T = 0.1
        iters = 0
        max_iters = 80000
        
        while time.time() < DEADLINE - 10 and T > min_T and iters < max_iters:
            iters += 1
            r = rng.random()
            
            if r < 0.22:
                # Row rebuild
                i = rng.randint(0, n_rows - 1)
                old_row = list(cur_rows[i])
                used_without = 0
                for ri in range(n_rows):
                    if ri != i:
                        used_without |= row_mask(cur_rows[ri])

                res = row_rebuild_greedy(used_without, cur_scope + 5, rng)
                if res is None:
                    continue
                new_row, new_mask = res

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                new_cost = get_cost(cur_rows, diff_counts)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope_sa = scope_of(cur_rows)
                        best_rows_sa = [list(rr) for rr in cur_rows]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row

            elif r < 0.40:
                # Structural swap move
                if try_swap_two_entries(cur_rows, diff_counts, rng):
                    new_cost = get_cost(cur_rows, diff_counts)
                    if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                        cur_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_scope_sa = scope_of(cur_rows)
                            best_rows_sa = [list(rr) for rr in cur_rows]
                    else:
                        pass

            elif r < 0.65:
                # Scope compression via DFS
                max_val = cur_scope
                max_row_idx = -1
                for ri in range(n_rows):
                    if cur_rows[ri][-1] == max_val:
                        max_row_idx = ri
                        break
                if max_row_idx < 0:
                    max_row_idx = rng.randint(0, n_rows - 1)
                    max_val = cur_rows[max_row_idx][-1]
                    
                i = max_row_idx
                old_row = list(cur_rows[i])
                used_without = 0
                for ri in range(n_rows):
                    if ri != i:
                        used_without |= row_mask(cur_rows[ri])

                dfs_dl = min(DEADLINE - 5, time.time() + 0.4)
                res = find_row_dfs(used_without, old_row[-1], old_row[-1] - 1, 6000, dfs_dl)
                if res is not None:
                    new_row, new_mask = res
                    remove_row_diffs(diff_counts, old_row)
                    add_row_diffs(diff_counts, new_row)
                    cur_rows[i] = new_row
                    new_cost = get_cost(cur_rows, diff_counts)

                    if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                        cur_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_scope_sa = scope_of(cur_rows)
                            best_rows_sa = [list(rr) for rr in cur_rows]
                    else:
                        remove_row_diffs(diff_counts, new_row)
                        add_row_diffs(diff_counts, old_row)
                        cur_rows[i] = old_row

            else:
                # Single element move
                i = rng.randint(0, n_rows - 1)
                j = rng.randint(1, row_len - 1)
                old_row = list(cur_rows[i])
                used_without = 0
                for ri in range(n_rows):
                    if ri != i:
                        used_without |= row_mask(cur_rows[ri])

                old_val = old_row[j]
                improved_this = False
                for delta in range(-4, 5):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    if j > 0 and new_val <= old_row[j - 1]:
                        continue
                    if j < len(old_row) - 1 and new_val >= old_row[j + 1]:
                        continue
                    
                    new_row = list(old_row)
                    new_row[j] = new_val
                    new_mask = row_mask(new_row)
                    
                    if (new_mask & used_without) == 0:
                        remove_row_diffs(diff_counts, old_row)
                        add_row_diffs(diff_counts, new_row)
                        cur_rows[i] = new_row
                        new_cost = get_cost(cur_rows, diff_counts)

                        if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                            cur_cost = new_cost
                            if new_cost < best_cost:
                                best_cost = new_cost
                                best_scope_sa = scope_of(cur_rows)
                                best_rows_sa = [list(rr) for rr in cur_rows]
                            improved_this = True
                            break
                        else:
                            remove_row_diffs(diff_counts, new_row)
                            add_row_diffs(diff_counts, old_row)
                            cur_rows[i] = old_row
                
                if not improved_this:
                    cur_rows[i] = old_row

            T *= cooling
        
        if best_scope_sa < best_scope:
            best_scope = best_scope_sa
            best_rows = best_rows_sa

    # Phase 3: Final local polish - try to reduce scope further
    if time.time() < DEADLINE - 2:
        improved = True
        while improved and time.time() < DEADLINE - 2:
            improved = False
            for i in range(n_rows):
                old_row = list(best_rows[i])
                old_max = old_row[-1]
                
                used_without = 0
                for ri in range(n_rows):
                    if ri != i:
                        used_without |= row_mask(best_rows[ri])
                
                dfs_dl = min(DEADLINE - 2, time.time() + 0.08)
                res = find_row_dfs(used_without, old_max, old_max - 1, 3000, dfs_dl)
                if res is not None:
                    new_row, _ = res
                    if new_row[-1] < old_max:
                        best_rows[i] = new_row
                        improved = True
                        break

    return best_rows
