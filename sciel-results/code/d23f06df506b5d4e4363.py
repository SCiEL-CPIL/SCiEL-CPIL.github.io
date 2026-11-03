"""Difference triangle set (7,5) solver: diversity greedy + gap-fill repair,
delta-tracked simulated annealing, and an exact downward-compression polish."""
import time
import random
import math

_DEADLINE = 143.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _sum_entries(rows):
    return sum(sum(r) for r in rows)

def _row_diffs(row):
    n = len(row)
    return [row[j] - row[i] for i in range(n) for j in range(i + 1, n)]

def _row_diffs_set(row):
    n = len(row)
    return {row[j] - row[i] for i in range(n) for j in range(i + 1, n)}

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

def _is_valid(rows, k):
    if not rows:
        return False
    seen = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _build_row_given_used(used, k, rng, cap=None, jitter=0.0):
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if cap is not None and x > cap:
            return None
        if jitter > 0 and rng.random() < jitter:
            x += rng.randint(1, 3)
            if cap is not None and x > cap:
                return None
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            row.append(x)
            used = used | new_diffs
    return row

def _fill_gaps(row, used, k, rng):
    """Inject the smallest globally-missing differences to force compression.
    Returns (new_row, new_used) or (None, used) if it cannot be done."""
    used = set(used)
    target = k * (k + 1) // 2
    cur = _row_diffs_set(row)
    row = list(row)
    while len(used) < target:
        missing = [d for d in range(1, max(used) + 2) if d not in used]
        if not missing:
            break
        best_d = None
        best_row = None
        best_cost = None
        for d in missing[:12]:
            new_row = None
            new_used = None
            for pos in range(1, len(row)):
                lo = row[pos - 1] + 1
                hi = row[pos + 1] - 1 if pos < len(row) - 1 else 10 ** 9
                for v in range(lo, hi + 1):
                    cand = list(row)
                    cand[pos] = v
                    s = _row_diffs_set(cand)
                    if len(s) != k * (k + 1) // 2:
                        continue
                    if d not in s:
                        continue
                    nu = used | s
                    if len(nu) == len(used) + 1:
                        cost = cand[-1]
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_d = d
                            best_row = cand
                            best_used = nu
            if best_row is not None and best_d == d:
                break
        if best_row is None:
            break
        row = best_row
        used = best_used
    return row, used

def _initial_construction(n_rows, k, rng, n_restarts=600):
    best = None
    best_score = float('inf')
    for attempt in range(n_restarts):
        if _elapsed() > _DEADLINE - 25:
            break
        used = set()
        rows = []
        ok = True
        for r in range(n_rows):
            div = 0.04 * (0.5 + rng.random())
            row = _build_row_given_used(used, k, rng, jitter=div)
            if row is None:
                ok = False
                break
            rows.append(row)
            used = used | _row_diffs_set(row)
        if not ok or len(rows) != n_rows:
            continue
        # Gap-fill repair pass to compress
        new_rows = []
        new_used = set()
        ok2 = True
        for row in rows:
            nr, nu = _fill_gaps(row, new_used, k, rng)
            if nr is None:
                ok2 = False
                break
            new_rows.append(nr)
            new_used = nu
        cand = new_rows if ok2 else rows
        if _is_valid(cand, k):
            sc = _max_entry(cand)
            if sc < best_score:
                best_score = sc
                best = [list(r) for r in cand]
    return best, best_score

def _sa_optimize(rows, k, rng, time_budget=100):
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    current_rows = [list(r) for r in best_rows]

    diff_counts = {}
    conflict_count = 0
    for row in current_rows:
        for d in _row_diffs(row):
            old_cnt = diff_counts.get(d, 0)
            if old_cnt >= 1:
                conflict_count += 1
            diff_counts[d] = old_cnt + 1

    current_scope = _max_entry(current_rows)
    current_sum = _sum_entries(current_rows)
    row_maxes = [r[-1] for r in current_rows]

    PENALTY = 500
    SUM_WEIGHT = 0.05
    current_cost = current_scope + PENALTY * conflict_count + SUM_WEIGHT * current_sum

    best_scope = current_scope
    best_conflicts = conflict_count
    best_sum = current_sum
    best_cost = current_cost

    start_time = time.time()
    temp = 12.0
    cooling = 0.9997

    row_diffs_list = [_row_diffs(row) for row in current_rows]
    row_diff_sets = [set(d) for d in row_diffs_list]

    def _other_max(exclude):
        m = 0
        for i in range(n_rows):
            if i != exclude and row_maxes[i] > m:
                m = row_maxes[i]
        return m

    def _accept_check(delta_cost):
        return delta_cost < 0 or (temp > 0.01 and rng.random() < math.exp(-delta_cost / temp))

    def _commit(new_rows, new_diffs_list, new_diff_sets, new_maxes,
                new_conf, new_scope, new_sum, new_cost):
        nonlocal current_rows, row_diffs_list, row_diff_sets, row_maxes
        nonlocal conflict_count, current_scope, current_sum, current_cost
        nonlocal best_cost, best_scope, best_conflicts, best_sum, best_rows
        current_rows = new_rows
        row_diffs_list = new_diffs_list
        row_diff_sets = new_diff_sets
        row_maxes = new_maxes
        conflict_count = new_conf
        current_scope = new_scope
        current_sum = new_sum
        current_cost = new_cost
        if new_cost < best_cost:
            best_cost = new_cost
            best_scope = new_scope
            best_conflicts = new_conf
            best_sum = new_sum
            best_rows = [list(r) for r in current_rows]

    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break

        move_type = rng.random()

        if move_type < 0.30:
            # Rebuild a random row greedily with a tight cap
            ri = rng.randrange(n_rows)
            old_row = current_rows[ri]
            old_diffs = row_diffs_list[ri]
            for d in old_diffs:
                c = diff_counts[d]
                if c >= 2:
                    conflict_count -= 1
                if c == 1:
                    del diff_counts[d]
                else:
                    diff_counts[d] = c - 1
            used = set(diff_counts.keys())
            cap = current_scope + rng.randint(0, 6)
            new_row = _build_row_given_used(used, k, rng, cap=cap, jitter=0.05)
            if new_row is None or len(new_row) < k + 1:
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                continue
            new_diffs = _row_diffs(new_row)
            for d in new_diffs:
                c = diff_counts.get(d, 0)
                if c >= 1:
                    conflict_count += 1
                diff_counts[d] = c + 1
            new_max = new_row[-1]
            new_scope = max(new_max, _other_max(ri))
            new_sum = current_sum + sum(new_row) - sum(old_row)
            new_cost = new_scope + PENALTY * conflict_count + SUM_WEIGHT * new_sum
            if _accept_check(new_cost - current_cost):
                cur2 = [list(r) for r in current_rows]
                cur2[ri] = new_row
                dl2 = [list(x) for x in row_diffs_list]
                dl2[ri] = new_diffs
                ds2 = [set(x) for x in row_diff_sets]
                ds2[ri] = set(new_diffs)
                mx2 = list(row_maxes)
                mx2[ri] = new_max
                _commit(cur2, dl2, ds2, mx2, conflict_count, new_scope, new_sum, new_cost)
            else:
                for d in new_diffs:
                    c = diff_counts[d]
                    if c >= 2:
                        conflict_count -= 1
                    if c == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = c - 1
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1

        elif move_type < 0.62:
            # Shift a single entry (bias toward compression)
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            if len(row) < 2:
                continue
            idx = rng.randint(1, len(row) - 1)
            old_val = row[idx]
            r = rng.random()
            if r < 0.6:
                new_val = old_val - rng.randint(1, 5)
            elif r < 0.85:
                new_val = old_val + rng.randint(-3, 3)
            else:
                new_val = old_val + rng.randint(-8, 8)
            if new_val <= 0 or new_val == old_val:
                continue
            if idx > 1 and new_val <= row[idx - 1]:
                continue
            if idx < len(row) - 1 and new_val >= row[idx + 1]:
                continue
            new_row = list(row)
            new_row[idx] = new_val
            new_diffs = _row_diffs(new_row)
            new_set = set(new_diffs)
            if len(new_diffs) != len(new_set):
                continue
            old_set = row_diff_sets[ri]
            removed = old_set - new_set
            added = new_set - old_set
            delta_conf = 0
            for d in removed:
                if diff_counts.get(d, 0) >= 2:
                    delta_conf -= 1
            for d in added:
                if diff_counts.get(d, 0) >= 1:
                    delta_conf += 1
            new_max = new_row[-1]
            new_scope = max(new_max, _other_max(ri))
            new_conf = conflict_count + delta_conf
            new_sum = current_sum + (new_val - old_val)
            new_cost = new_scope + PENALTY * new_conf + SUM_WEIGHT * new_sum
            if _accept_check(new_cost - current_cost):
                for d in removed:
                    c = diff_counts[d]
                    if c == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = c - 1
                for d in added:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                cur2 = [list(x) for x in current_rows]
                cur2[ri] = new_row
                dl2 = [list(x) for x in row_diffs_list]
                dl2[ri] = new_diffs
                ds2 = [set(x) for x in row_diff_sets]
                ds2[ri] = new_set
                mx2 = list(row_maxes)
                mx2[ri] = new_max
                _commit(cur2, dl2, ds2, mx2, new_conf, new_scope, new_sum, new_cost)

        elif move_type < 0.85:
            # Cheap scope move: pull the last entry of a row downward
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            if len(row) < 2:
                continue
            old_val = row[-1]
            lo = row[-2] + 1
            new_val = old_val - rng.randint(1, 4)
            if new_val < lo:
                new_val = lo
            if new_val >= old_val:
                continue
            new_row = list(row)
            new_row[-1] = new_val
            new_diffs = _row_diffs(new_row)
            new_set = set(new_diffs)
            if len(new_diffs) != len(new_set):
                continue
            old_set = row_diff_sets[ri]
            removed = old_set - new_set
            added = new_set - old_set
            delta_conf = 0
            for d in removed:
                if diff_counts.get(d, 0) >= 2:
                    delta_conf -= 1
            for d in added:
                if diff_counts.get(d, 0) >= 1:
                    delta_conf += 1
            new_scope = max(new_val, _other_max(ri))
            new_conf = conflict_count + delta_conf
            new_sum = current_sum + (new_val - old_val)
            new_cost = new_scope + PENALTY * new_conf + SUM_WEIGHT * new_sum
            if _accept_check(new_cost - current_cost):
                for d in removed:
                    c = diff_counts[d]
                    if c == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = c - 1
                for d in added:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                cur2 = [list(x) for x in current_rows]
                cur2[ri] = new_row
                dl2 = [list(x) for x in row_diffs_list]
                dl2[ri] = new_diffs
                ds2 = [set(x) for x in row_diff_sets]
                ds2[ri] = new_set
                mx2 = list(row_maxes)
                mx2[ri] = new_val
                _commit(cur2, dl2, ds2, mx2, new_conf, new_scope, new_sum, new_cost)

        else:
            # Swap last entries between two rows
            ri1 = rng.randrange(n_rows)
            ri2 = rng.randrange(n_rows)
            if ri1 == ri2:
                continue
            row1 = current_rows[ri1]
            row2 = current_rows[ri2]
            v1 = row1[-1]
            v2 = row2[-1]
            if v1 == v2:
                continue
            if len(row1) > 1 and v2 <= row1[-2]:
                continue
            if len(row2) > 1 and v1 <= row2[-2]:
                continue
            nr1 = list(row1); nr1[-1] = v2
            nr2 = list(row2); nr2[-1] = v1
            nd1 = _row_diffs(nr1); nd2 = _row_diffs(nr2)
            s1 = set(nd1); s2 = set(nd2)
            if len(nd1) != len(s1) or len(nd2) != len(s2):
                continue
            os1 = row_diff_sets[ri1]; os2 = row_diff_sets[ri2]
            removed = (os1 - s1) | (os2 - s2)
            added = (s1 - os1) | (s2 - os2)
            delta_conf = 0
            for d in removed:
                if diff_counts.get(d, 0) >= 2:
                    delta_conf -= 1
            for d in added:
                if diff_counts.get(d, 0) >= 1:
                    delta_conf += 1
            other_max = 0
            for i in range(n_rows):
                if i != ri1 and i != ri2 and row_maxes[i] > other_max:
                    other_max = row_maxes[i]
            new_scope = max(v1, v2, other_max)
            new_conf = conflict_count + delta_conf
            new_cost = new_scope + PENALTY * new_conf + SUM_WEIGHT * current_sum
            if _accept_check(new_cost - current_cost):
                for d in removed:
                    c = diff_counts[d]
                    if c == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = c - 1
                for d in added:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                cur2 = [list(x) for x in current_rows]
                cur2[ri1] = nr1
                cur2[ri2] = nr2
                dl2 = [list(x) for x in row_diffs_list]
                dl2[ri1] = nd1
                dl2[ri2] = nd2
                ds2 = [set(x) for x in row_diff_sets]
                ds2[ri1] = s1
                ds2[ri2] = s2
                mx2 = list(row_maxes)
                mx2[ri1] = v2
                mx2[ri2] = v1
                _commit(cur2, dl2, ds2, mx2, new_conf, new_scope, current_sum, new_cost)

        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    if _is_valid(best_rows, k):
        return best_rows
    if _is_valid(current_rows, k):
        return current_rows
    return best_rows

def _local_search_polish(rows, k, rng, time_budget=15):
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    best_sum = _sum_entries(best_rows)

    diff_counts = {}
    for row in best_rows:
        for d in _row_diffs(row):
            diff_counts[d] = diff_counts.get(d, 0) + 1
    row_diff_sets = [_row_diffs_set(row) for row in best_rows]
    row_diff_lists = [_row_diffs(row) for row in best_rows]
    row_maxes = [r[-1] for r in best_rows]

    def _other_max(exclude):
        m = 0
        for i in range(n_rows):
            if i != exclude and row_maxes[i] > m:
                m = row_maxes[i]
        return m

    start_time = time.time()
    no_improve = 0
    MAX_NO_IMPROVE = 8

    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 3:
            break
        improved = False
        for ri in range(n_rows):
            row = best_rows[ri]
            row_len = len(row)
            for idx in range(1, row_len):
                old_val = row[idx]
                min_val = row[idx - 1] + 1 if idx > 1 else 1
                max_val = row[idx + 1] - 1 if idx < row_len - 1 else old_val + 2
                for reduction in range(1, min(6, old_val - min_val + 1)):
                    new_val = old_val - reduction
                    if new_val < min_val:
                        break
                    if idx < row_len - 1 and new_val >= row[idx + 1]:
                        break
                    new_row = list(row)
                    new_row[idx] = new_val
                    new_diffs = _row_diffs(new_row)
                    new_set = set(new_diffs)
                    if len(new_diffs) != len(new_set):
                        continue
                    old_set = row_diff_sets[ri]
                    removed = old_set - new_set
                    added = new_set - old_set
                    conflict = False
                    for d in added:
                        oc = diff_counts.get(d, 0)
                        if d in old_set:
                            oc -= 1
                        if oc >= 1:
                            conflict = True
                            break
                    if conflict:
                        continue
                    new_max = new_row[-1]
                    new_scope = max(new_max, _other_max(ri))
                    new_sum = best_sum + (new_val - old_val)
                    if new_scope < best_scope or (new_scope == best_scope and new_sum < best_sum):
                        for d in removed:
                            c = diff_counts[d]
                            if c == 1:
                                del diff_counts[d]
                            else:
                                diff_counts[d] = c - 1
                        for d in added:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        row[idx] = new_val
                        row_diff_sets[ri] = new_set
                        row_diff_lists[ri] = new_diffs
                        row_maxes[ri] = new_max
                        best_scope = new_scope
                        best_sum = new_sum
                        improved = True
                        break
        if not improved:
            no_improve += 1
            if no_improve >= MAX_NO_IMPROVE:
                break
        else:
            no_improve = 0

    # Final deterministic downward sweep: pull every entry down as far as possible
    changed = True
    while changed and _elapsed() < _DEADLINE - 3:
        changed = False
        for ri in range(n_rows):
            row = best_rows[ri]
            row_len = len(row)
            for idx in range(row_len - 1, 0, -1):
                old_val = row[idx]
                min_val = row[idx - 1] + 1
                max_val = row[idx + 1] - 1 if idx < row_len - 1 else old_val
                for new_val in range(max_val, min_val - 1, -1):
                    if new_val >= old_val:
                        continue
                    new_row = list(row)
                    new_row[idx] = new_val
                    new_diffs = _row_diffs(new_row)
                    new_set = set(new_diffs)
                    if len(new_diffs) != len(new_set):
                        continue
                    old_set = row_diff_sets[ri]
                    removed = old_set - new_set
                    added = new_set - old_set
                    conflict = False
                    for d in added:
                        oc = diff_counts.get(d, 0)
                        if d in old_set:
                            oc -= 1
                        if oc >= 1:
                            conflict = True
                            break
                    if conflict:
                        continue
                    new_max = new_row[-1]
                    new_scope = max(new_max, _other_max(ri))
                    new_sum = best_sum + (new_val - old_val)
                    if new_scope < best_scope or (new_scope == best_scope and new_sum < best_sum):
                        for d in removed:
                            c = diff_counts[d]
                            if c == 1:
                                del diff_counts[d]
                            else:
                                diff_counts[d] = c - 1
                        for d in added:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        row[idx] = new_val
                        row_diff_sets[ri] = new_set
                        row_diff_lists[ri] = new_diffs
                        row_maxes[ri] = new_max
                        best_scope = new_scope
                        best_sum = new_sum
                        changed = True
                        break

    if _is_valid(best_rows, k):
        return best_rows
    return [list(r) for r in rows]

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()

    rng = random.Random(42)

    best_rows, best_score = _initial_construction(n_rows, k, rng)

    if best_rows is None:
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
        best_score = _max_entry(best_rows)

    if _elapsed() < _DEADLINE - 18:
        time_left = _DEADLINE - 18 - _elapsed()
        sa_time = min(115, time_left)
        improved_rows = _sa_optimize(best_rows, k, rng, time_budget=sa_time)
        if improved_rows and _is_valid(improved_rows, k):
            ns = _max_entry(improved_rows)
            if ns < best_score:
                best_rows = improved_rows
                best_score = ns

    if _elapsed() < _DEADLINE - 3:
        polish_time = min(18, _DEADLINE - 3 - _elapsed())
        if polish_time > 0:
            polished_rows = _local_search_polish(best_rows, k, rng, time_budget=polish_time)
            if polished_rows and _is_valid(polished_rows, k):
                ns = _max_entry(polished_rows)
                if ns < best_score:
                    best_rows = polished_rows
                    best_score = ns

    if not _is_valid(best_rows, k):
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)

    return best_rows
