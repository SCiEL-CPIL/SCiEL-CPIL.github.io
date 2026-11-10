import time
import random

def construct(n_rows, k):
    start = time.time()
    deadline = start + 130  # safety margin before 150s wall-clock

    def row_diffs(row):
        out = set()
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - a)
        return out

    def greedy_row(used_set, k, cap=None):
        """Smallest-next-mark row avoiding used_set. Returns None if cap too tight."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set = used_set | new
        return row

    def greedy_row_random(used_set, k, cap=None, skip_prob=0.1):
        """Greedy with random skips for diversity."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            # Randomly skip some candidates
            if random.random() < skip_prob and len(row) < k:
                continue
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set = used_set | new
        return row

    def dfs_row(row, used_set, cap, row_deadline):
        """Find a row with last element <= cap avoiding used_set. Returns row or None."""
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k + 1:
            return row[:]
        last = row[-1]
        needed = (k + 1) - len(row)
        upper = cap - (needed - 1)
        if upper < last + 1:
            return None
        for x in range(last + 1, upper + 1):
            if time.time() > row_deadline or time.time() > deadline:
                return None
            new = {x - a for a in row}
            if new & used_set:
                continue
            res = dfs_row(row + [x], used_set | new, cap, row_deadline)
            if res is not None:
                return res
        return None

    def dfs_row_backtrack(row, used_set, cap, row_deadline, max_depth=3):
        """DFS with limited backtracking for better search."""
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k + 1:
            return row[:]
        last = row[-1]
        needed = (k + 1) - len(row)
        upper = cap - (needed - 1)
        if upper < last + 1:
            return None
        
        # Try candidates in order, with limited backtracking
        candidates = list(range(last + 1, upper + 1))
        
        def _dfs(r, us, d):
            if time.time() > row_deadline or time.time() > deadline:
                return None
            if len(r) == k + 1:
                return r[:]
            last_v = r[-1]
            need = (k + 1) - len(r)
            up = cap - (need - 1)
            if up < last_v + 1:
                return None
            for x in range(last_v + 1, up + 1):
                if time.time() > row_deadline or time.time() > deadline:
                    return None
                new = {x - a for a in r}
                if new & us:
                    continue
                res = _dfs(r + [x], us | new, d + 1)
                if res is not None:
                    return res
            return None
        
        return _dfs(row, used_set, 0)

    # ---- Phase 1: Multiple greedy baselines, keep the best ----
    best_rows = None
    best_max = float('inf')
    best_used = None
    
    for attempt in range(5):
        if time.time() > deadline - 60:
            break
        used = set()
        rows = []
        for i in range(n_rows):
            # Mix standard and random greedy
            if attempt == 0 or i % 3 == 0:
                row = greedy_row(set(used), k)
            else:
                row = greedy_row_random(set(used), k, skip_prob=0.05 + 0.02 * attempt)
                if row is None:
                    row = greedy_row(set(used), k)
            rows.append(row)
            used |= row_diffs(row)
        
        cur_max = max(row[-1] for row in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows
            best_used = used
    
    rows = best_rows
    used = best_used

    # ---- Phase 2: Per-row DFS refinement during construction ----
    for i in range(n_rows):
        if time.time() > deadline - 30:
            break
        cur = rows[i]
        cur_last = cur[-1]
        best = None
        # Try to find a smaller row for position i
        for cap in range(cur_last - 1, max(cur_last - 50, 0), -1):
            if time.time() > deadline - 30:
                break
            used_without = set(used)
            for j in range(i, n_rows):
                used_without.difference_update(row_diffs(rows[j]))
            row_deadline = min(deadline - 30, time.time() + 1.5)
            cand = dfs_row([0], used_without, cap, row_deadline)
            if cand is not None:
                best = cand
                break
        if best is not None and best[-1] < cur_last:
            for j in range(i, n_rows):
                used.difference_update(row_diffs(rows[j]))
            used |= row_diffs(best)
            rows[i] = best
            for j in range(i + 1, n_rows):
                row = greedy_row(set(used), k)
                rows[j] = row
                used |= row_diffs(row)

    # ---- Phase 3: Iterative improvement - replace rows with smaller ones ----
    order = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for r in order:
            if time.time() > deadline - 20:
                break
            cur_last = rows[r][-1]
            best = None
            for cap in range(cur_last - 1, max(cur_last - 40, 0), -1):
                if time.time() > deadline - 20:
                    break
                used_without = set(used)
                used_without.difference_update(row_diffs(rows[r]))
                row_deadline = min(deadline - 20, time.time() + 1.5)
                cand = dfs_row([0], used_without, cap, row_deadline)
                if cand is not None:
                    best = cand
                    break
            if best is not None and best[-1] < cur_last:
                used.difference_update(row_diffs(rows[r]))
                rows[r] = best
                used |= row_diffs(best)
                improved = True
                # Re-sort since max changed
                order = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)

    # ---- Phase 4: Local search - reduce max entry within a row ----
    if time.time() < deadline - 15:
        for r in range(n_rows):
            if time.time() > deadline - 15:
                break
            row = rows[r]
            cur_max = row[-1]
            used_without = set(used)
            used_without.difference_update(row_diffs(row))

            improved_row = None
            for attempt in range(3):
                if time.time() > deadline - 15:
                    break
                cand_row = row[:]
                # Try to reduce elements from the end
                for pos in range(len(cand_row) - 2, -1, -1):
                    if time.time() > deadline - 15:
                        break
                    old_val = cand_row[pos]
                    lower = cand_row[pos-1] + 1 if pos > 0 else 0
                    for new_val in range(old_val - 1, lower - 1, -1):
                        if time.time() > deadline - 15:
                            break
                        test_row = cand_row[:]
                        test_row[pos] = new_val
                        test_diffs = row_diffs(test_row)
                        if not (test_diffs & used_without):
                            if all(test_row[i] < test_row[i+1] for i in range(len(test_row)-1)):
                                cand_row = test_row
                                break
                if cand_row[-1] < cur_max:
                    improved_row = cand_row
                    break

            if improved_row is not None and improved_row[-1] < cur_max:
                used.difference_update(row_diffs(rows[r]))
                rows[r] = improved_row
                used |= row_diffs(improved_row)

    # ---- Phase 5: Row-swap local search with randomized replacement ----
    if time.time() < deadline - 10:
        for _ in range(3):  # Multiple passes
            if time.time() > deadline - 10:
                break
            improved_swap = True
            while improved_swap and time.time() < deadline - 10:
                improved_swap = False
                for i in range(n_rows):
                    if time.time() > deadline - 10:
                        break
                    for j in range(i + 1, n_rows):
                        if time.time() > deadline - 10:
                            break
                        # Try swapping rows i and j, then see if we can improve
                        # First, just check if swapping helps
                        row_i = rows[i]
                        row_j = rows[j]
                        if row_i[-1] == row_j[-1]:
                            continue
                        
                        # Try to find a better replacement for the larger row
                        larger = i if row_i[-1] > row_j[-1] else j
                        smaller = j if larger == i else i
                        cur_last = rows[larger][-1]
                        
                        used_without = set(used)
                        used_without.difference_update(row_diffs(rows[larger]))
                        
                        best = None
                        for cap in range(cur_last - 1, max(cur_last - 30, 0), -1):
                            if time.time() > deadline - 10:
                                break
                            row_deadline = min(deadline - 10, time.time() + 1.0)
                            cand = dfs_row([0], used_without, cap, row_deadline)
                            if cand is not None:
                                best = cand
                                break
                        
                        if best is not None and best[-1] < cur_last:
                            used.difference_update(row_diffs(rows[larger]))
                            rows[larger] = best
                            used |= row_diffs(best)
                            improved_swap = True
                            break
                    if improved_swap:
                        break

    # ---- Phase 6: Randomized restarts with perturbation ----
    best_rows_final = rows[:]
    best_max_final = max(row[-1] for row in rows)

    if time.time() < deadline - 15:
        for restart in range(4):
            if time.time() > deadline - 15:
                break
            used_r = set()
            rows_r = []
            for i in range(n_rows):
                # Try multiple random constructions, keep the best
                best_row = None
                best_last = float('inf')
                for _ in range(4):
                    if random.random() < 0.5:
                        row = greedy_row(set(used_r), k)
                    else:
                        row = greedy_row_random(set(used_r), k, skip_prob=0.05 + 0.03 * restart)
                    if row is not None and row[-1] < best_last:
                        best_row = row
                        best_last = row[-1]
                if best_row is None:
                    best_row = greedy_row(set(used_r), k)
                rows_r.append(best_row)
                used_r |= row_diffs(best_row)

            cur_max = max(row[-1] for row in rows_r)
            if cur_max < best_max_final:
                best_rows_final = rows_r
                best_max_final = cur_max
                used = used_r
                rows = rows_r

                # Quick improvement pass on new best
                if time.time() < deadline - 10:
                    order2 = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
                    improved2 = True
                    while improved2 and time.time() < deadline - 10:
                        improved2 = False
                        for r in order2:
                            if time.time() > deadline - 10:
                                break
                            cur_last = rows[r][-1]
                            best = None
                            for cap in range(cur_last - 1, max(cur_last - 25, 0), -1):
                                if time.time() > deadline - 10:
                                    break
                                used_without = set(used)
                                used_without.difference_update(row_diffs(rows[r]))
                                row_deadline = min(deadline - 10, time.time() + 1.0)
                                cand = dfs_row([0], used_without, cap, row_deadline)
                                if cand is not None:
                                    best = cand
                                    break
                            if best is not None and best[-1] < cur_last:
                                used.difference_update(row_diffs(rows[r]))
                                rows[r] = best
                                used |= row_diffs(best)
                                improved2 = True
                                order2 = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)

    return best_rows_final
