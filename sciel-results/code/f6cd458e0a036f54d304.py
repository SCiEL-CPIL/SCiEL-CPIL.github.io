"""Combined program: greedy construction + randomized restarts + simulated-annealing
refinement, all guarded by an explicit wall-clock deadline."""
import time
import random
import math

_START = time.time()
_DEADLINE = _START + 115.0   # evaluator kills at 150s; keep a safety margin


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    """All positive differences a_j - a_i for i<j in one row."""
    d = []
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            d.append(row[j] - ai)
    return d


def _row_diff_set(row):
    s = set()
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            s.add(row[j] - ai)
    return s


def _scope(rows):
    return max(max(r) for r in rows)


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            if r[i] >= r[i + 1]:
                return False
        for d in _row_diff_set(r):
            if d in used:
                return False
            used.add(d)
    return True


def _greedy_row(used, k, max_mark, rnd, jitter=0):
    """Place one row of length k+1 starting at 0 using the smallest marks whose
    differences are all unused. Returns (row, diffset, ok)."""
    row = [0]
    local = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        if x > max_mark:
            return None, local, False
        # occasional jitter: skip a few candidates to vary the row
        if jitter and rnd.random() < 0.25:
            x += rnd.randint(1, jitter)
            if x > max_mark:
                return None, local, False
        new = []
        bad = False
        for a in row:
            dv = x - a
            if dv in used or dv in local:
                bad = True
                break
            new.append(dv)
        if not bad:
            row.append(x)
            local.update(new)
    return row, local, True


def _greedy_fill(n_rows, k, max_mark, rnd, jitter=0):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, local, ok = _greedy_row(used, k, max_mark, rnd, jitter)
        if not ok:
            return None, None
        rows.append(row)
        used |= local
    return rows, used


def _regrow_row(used, k, max_mark, rnd, keep_idx, cur_row):
    """Replace row keep_idx with a fresh greedy row, leaving all other rows'
    differences in `used`. Returns (new_row, ok)."""
    other_used = set(used)
    # remove cur_row's own diffs so it can be re-placed
    for d in _row_diff_set(cur_row):
        other_used.discard(d)
    row, local, ok = _greedy_row(other_used, k, max_mark, rnd)
    if not ok:
        return None, False
    return row, True


def _local_search(rows, n_rows, k, rnd, time_budget):
    """Simulated-annealing style: repeatedly replace a row with a new greedy row
    that keeps the packing valid, accepting improvements (and occasional
    equal-scoring swaps) to reduce the scope. Returns best rows found."""
    cur = [list(r) for r in rows]
    cur_scope = _scope(cur)
    best = [list(r) for r in cur]
    best_scope = cur_scope
    used = set()
    for r in cur:
        used.update(_row_diff_set(r))

    T = 6.0
    cooldown = 0.9995
    T_min = 0.01
    while not _dead() and (time.time() - _START) < time_budget:
        T *= cooldown
        if T < T_min:
            T = T_min
        for _ in range(3000):
            if _dead() or (time.time() - _START) >= time_budget:
                break
            ri = rnd.randrange(n_rows)
            # try to regrow row ri at the current scope
            new_row, ok = _regrow_row(used, k, cur_scope, rnd, ri, cur[ri])
            if not ok:
                continue
            # compute new scope
            new_scope = max(cur_scope, max(new_row))
            delta = new_scope - cur_scope
            # accept if improves, or equal, or (rarely) slightly worse when hot
            if delta < 0 or (delta == 0 and rnd.random() < 0.3) or (delta > 0 and rnd.random() < math.exp(-delta / T)):
                # update used set
                for d in _row_diff_set(cur[ri]):
                    used.discard(d)
                cur[ri] = new_row
                for d in _row_diff_set(new_row):
                    used.add(d)
                cur_scope = new_scope
                if cur_scope < best_scope:
                    best_scope = cur_scope
                    best = [list(r) for r in cur]
    return best, best_scope


def construct(n_rows, k):
    rnd = random.Random(12345)
    best_rows = None
    best_scope = None

    # Counting lower bound on scope
    lb = n_rows * k * (k + 1) // 2

    # ---- Phase A: randomized greedy packing, find a feasible scope ----
    for m in range(lb, lb + 3000):
        if _dead():
            break
        found = False
        for attempt in range(80):
            if _dead():
                break
            jitter = attempt % 4   # vary the search a bit
            rows, used = _greedy_fill(n_rows, k, m, rnd, jitter)
            if rows is not None and _valid(rows, n_rows, k):
                sc = _scope(rows)
                if best_scope is None or sc < best_scope:
                    best_scope = sc
                    best_rows = rows
                found = True
                break
        if found:
            # probe slightly lower scopes for a lucky packing
            for mm in range(max(lb, m - 20), m):
                if _dead():
                    break
                for attempt in range(20):
                    if _dead():
                        break
                    rows, used = _greedy_fill(n_rows, k, mm, rnd)
                    if rows is not None and _valid(rows, n_rows, k):
                        sc = _scope(rows)
                        if best_scope is None or sc < best_scope:
                            best_scope = sc
                            best_rows = rows
            break

    # ---- Phase B: local search to shrink the scope ----
    if best_rows is not None and not _dead():
        time_budget = _DEADLINE - (time.time() - _START)
        if time_budget > 3.0:
            best_rows, best_scope = _local_search(best_rows, n_rows, k, rnd, time_budget)

    if best_rows is None:
        # Fallback: pure greedy with a generous scope
        rows, used = _greedy_fill(n_rows, k, lb + 5000, rnd)
        if rows is None:
            rows = []
            used = set()
            for _ in range(n_rows):
                row, local, _ = _greedy_row(used, k, 1000000, rnd)
                rows.append(row)
                used |= local
        return rows

    return best_rows
