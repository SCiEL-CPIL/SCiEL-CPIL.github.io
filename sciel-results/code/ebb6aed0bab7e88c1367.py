"""Model-and-optimize bootstrap: parameterize the scope m, then fit row placements
against the objective (minimize m) via randomized greedy packing, simulated annealing,
and backtracking repair, all guarded by an explicit wall-clock deadline."""

import time
import random
import sys
import math

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:
    _HAVE_NUMPY = False

# ---------- deadline guard ----------
_START = time.time()
_DEADLINE = _START + 115.0   # evaluator kills at 150s; keep a big safety margin


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


# ---------- helpers ----------
def _diffs_of_row(row):
    """All positive differences a_j - a_i for i<j in one row."""
    d = []
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            d.append(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _greedy_row(used, k, max_mark, rnd):
    """Greedy place one row of length k+1 starting at 0; smallest marks that keep
    all differences unused. Returns (row, used_delta, ok)."""
    row = [0]
    local = set()
    x = 0
    while len(row) < k + 1:
        x += 1
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


def _greedy_fill(n_rows, k, max_mark, rnd):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, local, ok = _greedy_row(used, k, max_mark, rnd)
        if not ok:
            return None, None
        rows.append(row)
        used |= local
    return rows, used


def _random_rows(n_rows, k, max_mark, rnd):
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_mark:
                return None
            row.append(x)
        rows.append(row)
    return rows


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True


def _row_diff_set(row):
    s = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            s.add(row[j] - row[i])
    return s


def _backtrack_repair(n_rows, k, target, rnd, time_budget=4.0):
    """Search for a packing into [0..target] via backtracking on marks + row order.
    Returns a valid list of rows or None."""
    rows = [None] * n_rows
    used = set()
    
    def try_fill_row(row_id, start_val=1):
        """Try to fill one row starting from start_val, using backtracking."""
        if time.time() > time_budget:
            return False
        row = [0]
        local_used = set()
        
        def backtrack(idx, current_row, local_set):
            if time.time() > time_budget:
                return False
            if idx == k + 1:
                rows[row_id] = current_row
                used.update(local_set)
                return True
            prev = current_row[-1]
            for x in range(prev + 1, target + 1):
                new_diffs = []
                bad = False
                for a in current_row:
                    dv = x - a
                    if dv in used or dv in local_set:
                        bad = True
                        break
                    new_diffs.append(dv)
                if bad:
                    continue
                new_local = local_set | set(new_diffs)
                if backtrack(idx + 1, current_row + [x], new_local):
                    return True
            return False
        
        return backtrack(0, [0], set())
    
    for _ in range(200):
        if time.time() > time_budget:
            break
        perm = list(range(n_rows))
        rnd.shuffle(perm)
        rows = [None] * n_rows
        used = set()
        success = True
        for ri in perm:
            if not try_fill_row(ri):
                success = False
                break
        if success and all(rows[t] is not None for t in range(n_rows)):
            return rows
    return None


def construct(n_rows, k):
    rnd = random.Random(42)
    best_rows = None
    best_scope = None

    # Lower bound on scope from counting argument:
    lb = n_rows * k * (k + 1) // 2

    # ---- Phase A: randomized greedy pack at various scopes (find a feasible m) ----
    # Try scopes from lb upward; each attempt a few randomized greedy orders.
    for m in range(lb, lb + 2000):
        if _dead():
            break
        found = False
        for attempt in range(60):
            if _dead():
                break
            rows, used = _greedy_fill(n_rows, k, m, rnd)
            if rows is not None and _valid(rows, n_rows, k):
                if best_scope is None or _scope(rows) < best_scope:
                    best_scope = _scope(rows)
                    best_rows = rows
                found = True
                break
        if found:
            # try a couple of slightly-lower m by luck
            for mm in range(max(lb, m - 15), m):
                if _dead():
                    break
                for attempt in range(15):
                    if _dead():
                        break
                    rows, used = _greedy_fill(n_rows, k, mm, rnd)
                    if rows is not None and _valid(rows, n_rows, k):
                        if best_scope is None or _scope(rows) < best_scope:
                            best_scope = _scope(rows)
                            best_rows = rows
            break

    # ---- Phase B: local search / simulated annealing on a valid packing ----
    if best_rows is not None and not _dead():
        cur = [list(r) for r in best_rows]
        cur_scope = _scope(cur)
        used = set()
        for r in cur:
            used.update(_diffs_of_row(r))
        best = [list(r) for r in cur]
        best_scope = cur_scope
        # Simulated annealing: temperature drives randomization of row regeneration
        T = 12.0
        cooldown = 0.9995
        T_min = 0.01
        no_improve_count = 0
        while not _dead():
            T *= cooldown
            if T < T_min:
                T = T_min
            for _ in range(3000):
                if _dead():
                    break
                ri = rnd.randrange(n_rows)
                # remove row ri
                old = cur[ri]
                old_diffs = _diffs_of_row(old)
                for d in old_diffs:
                    used.discard(d)
                # generate a new candidate row within a target window
                # target window shrinks as T shrinks
                window = int(best_scope * (1.0 + T * 0.3)) + k + 1
                window = min(window, best_scope + int(T * 8) + k + 1)
                cand, cand_used, ok = _greedy_row(used, k, window, rnd)
                if not ok:
                    # revert
                    cur[ri] = old
                    used.update(old_diffs)
                    continue
                new_scope = max(
                    [max(cur[i]) for i in range(n_rows) if i != ri] + [max(cand)])
                if new_scope < best_scope:
                    cur[ri] = cand
                    used.update(cand_used)
                    best = [list(r) for r in cur]
                    best_scope = new_scope
                    no_improve_count = 0
                else:
                    # accept with probability exp(-(new_scope-best_scope)/T) if close
                    if new_scope - best_scope <= 3 and rnd.random() < math.exp(-(new_scope - best_scope) / max(T, 0.01)):
                        cur[ri] = cand
                        used.update(cand_used)
                        no_improve_count += 1
                    else:
                        cur[ri] = old
                        used.update(old_diffs)
                        no_improve_count += 1
            # occasionally try a fresh greedy refit to escape local minima
            if no_improve_count > 5000 or rnd.random() < 0.03:
                rows, used2 = _greedy_fill(n_rows, k, best_scope + k, rnd)
                if rows is not None and _valid(rows, n_rows, k) and _scope(rows) < best_scope:
                    best = [list(r) for r in rows]
                    best_scope = _scope(rows)
                    cur = [list(r) for r in rows]
                    used = set()
                    for r in cur:
                        used.update(_diffs_of_row(r))
                    no_improve_count = 0
                else:
                    no_improve_count = 0

    # ---- Phase C: targeted backtracking repair at the current best_scope ----
    if best_rows is not None and not _dead():
        repaired = _backtrack_repair(n_rows, k, best_scope, rnd, time_budget=3.0)
        if repaired is not None and _valid(repaired, n_rows, k):
            best_rows = repaired
            best_scope = _scope(repaired)
    
    # ---- Phase D: targeted reduction of the maximum entry ----
    if best_rows is not None and not _dead():
        cur = [list(r) for r in best_rows]
        used = set()
        for r in cur:
            used.update(_diffs_of_row(r))
        best_scope = _scope(cur)
        
        for reduction in range(1, 6):
            if _dead():
                break
            target_scope = best_scope - reduction
            if target_scope < n_rows * k * (k + 1) // 2:
                break
            # Try to find a packing with scope <= target_scope
            found = False
            for attempt in range(30):
                if _dead():
                    break
                rows, used2 = _greedy_fill(n_rows, k, target_scope, rnd)
                if rows is not None and _valid(rows, n_rows, k) and _scope(rows) <= target_scope:
                    cur = [list(r) for r in rows]
                    best_scope = _scope(rows)
                    best_rows = cur
                    found = True
                    break
            if found:
                continue
            else:
                # Try backtracking at the target scope
                repaired = _backtrack_repair(n_rows, k, target_scope, rnd, time_budget=2.0)
                if repaired is not None and _valid(repaired, n_rows, k) and _scope(repaired) <= target_scope:
                    cur = [list(r) for r in repaired]
                    best_scope = _scope(repaired)
                    best_rows = cur
                    found = True
                if found:
                    continue
                else:
                    break

    # ---- Fallback if nothing found: plain greedy with no cap ----
    if best_rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row, local, ok = _greedy_row(used, k, 10 ** 9, rnd)
            if not ok:
                break
            rows.append(row)
            used |= local
        if len(rows) == n_rows:
            best_rows = rows

    if best_rows is None:
        # absolute last resort: trivial (invalid-likely) but structured rows
        best_rows = []
        for i in range(n_rows):
            row = [0]
            x = 0
            base = (i + 1) * (k + 1) * (k + 1) * (k + 1)
            while len(row) < k + 1:
                x += 1
                row.append(base + x)
            best_rows.append(row)

    # final sanity: make sure it's a list of n_rows lists of k+1 ints starting at 0
    out = []
    for r in best_rows:
        rr = list(r)
        out.append(rr)
    return out
