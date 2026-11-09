"""Difference triangle set (7,5) – combined greedy + simulated annealing + backtracking."""
import time
import random
import sys
import math

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:
    _HAVE_NUMPY = False

_START = time.time()
_DEADLINE = _START + 142.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    """All positive differences a_j - a_i for i<j in one row."""
    d = set()
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            d.add(row[j] - ai)
    return d


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
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True


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


def _interleaved_greedy(n_rows, k, max_mark, rnd=None):
    """Build rows interleaved: at each step, extend the row with the smallest
    current maximum. This tends to keep all rows balanced and compact."""
    rows = [[0] for _ in range(n_rows)]
    used = set()

    for _ in range(n_rows * k):
        candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: rows[idx][-1])
        row = rows[i]
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
                found = True
                break
            x += 1
        if not found:
            return None, None

    for r in rows:
        if len(r) != k + 1:
            return None, None
    return rows, used


def _simulated_annealing(rows, n_rows, k, time_budget, rnd):
    """Simulated annealing to reduce scope. Moves: replace one row with a new
    random/greedy row that keeps the packing valid."""
    if rows is None:
        return None, None

    best = [r[:] for r in rows]
    best_scope = _scope(best)
    current = [r[:] for r in best]
    current_scope = best_scope

    # Precompute used set
    def compute_used(rows):
        used = set()
        for r in rows:
            used |= _diffs_of_row(r)
        return used

    used = compute_used(current)

    deadline = time.time() + time_budget
    T = 50.0
    cooling = 0.995
    step = 0

    while time.time() < deadline and T > 0.1:
        step += 1
        if step % 1000 == 0 and _dead():
            break

        # Pick a random row to replace
        ri = rnd.randint(0, n_rows - 1)
        old_row = current[ri]

        # Generate a new row: try greedy with slight randomization
        # Remove old row's diffs from used
        old_diffs = _diffs_of_row(old_row)
        used_without = used - old_diffs

        # Try to build a new row with max_mark = current_scope - 1 (or best_scope - 1)
        max_mark = min(current_scope, best_scope) - 1
        if max_mark < k:
            max_mark = best_scope

        # Try multiple randomizations
        new_row = None
        for attempt in range(20):
            # Slightly randomized greedy
            row = [0]
            local = set()
            x = 0
            success = True
            while len(row) < k + 1:
                x += 1
                if x > max_mark:
                    success = False
                    break
                # Sometimes skip a value to randomize
                if rnd.random() < 0.1 and x < max_mark:
                    x += 1
                    if x > max_mark:
                        success = False
                        break
                new = []
                bad = False
                for a in row:
                    dv = x - a
                    if dv in used_without or dv in local:
                        bad = True
                        break
                    new.append(dv)
                if not bad:
                    row.append(x)
                    local.update(new)
            if success and len(row) == k + 1:
                new_row = row
                break

        if new_row is None:
            # Try a completely random row
            row = [0]
            x = 0
            success = True
            while len(row) < k + 1:
                x += 1
                if x > max_mark:
                    success = False
                    break
                new = []
                bad = False
                for a in row:
                    dv = x - a
                    if dv in used_without or dv in local:
                        bad = True
                        break
                    new.append(dv)
                if not bad:
                    row.append(x)
                    local.update(new)
            if success and len(row) == k + 1:
                new_row = row

        if new_row is None:
            T *= cooling
            continue

        # Check validity of new row
        new_diffs = _diffs_of_row(new_row)
        if new_diffs & used_without:
            T *= cooling
            continue

        # Compute new scope
        new_scope = max(max(r) for i, r in enumerate(current) if i != ri)
        new_scope = max(new_scope, max(new_row))

        # Acceptance criterion
        delta = new_scope - current_scope
        if delta <= 0 or rnd.random() < math.exp(-delta / T):
            # Accept
            used = used_without | new_diffs
            current[ri] = new_row
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best = [r[:] for r in current]
        else:
            # Reject
            pass

        T *= cooling

    return best, best_scope


def _backtrack_search(n_rows, k, target, rnd, time_budget=3.0):
    """Backtracking search for a valid packing into [0..target]."""
    rows = [None] * n_rows
    used = set()

    def fill_row(row_id, marks, used_set):
        """Recursively fill one row."""
        if time.time() > time_budget:
            return False
        if len(marks) == k + 1:
            rows[row_id] = [0] + list(marks)
            return True
        start = (marks[-1] + 1) if marks else 1
        for x in range(start, target + 1):
            prev = [0] + list(marks)
            newl = set()
            bad = False
            for a in prev:
                dv = x - a
                if dv in used_set or dv in newl:
                    bad = True
                    break
                newl.add(dv)
            if bad:
                continue
            marks.append(x)
            used_set |= newl
            if fill_row(row_id, marks, used_set):
                return True
            used_set -= newl
            marks.pop()
        return False

    # Try multiple random orderings
    for attempt in range(50):
        if time.time() > time_budget:
            break
        perm = list(range(n_rows))
        rnd.shuffle(perm)
        rows = [None] * n_rows
        used = set()
        success = True
        for ri in perm:
            marks = []
            if not fill_row(ri, marks, used):
                success = False
                break
        if success and all(rows[t] is not None for t in range(n_rows)):
            return rows
    return None


def construct(n_rows, k):
    rnd = random.Random(42)
    best_rows = None
    best_scope = float('inf')

    # Counting lower bound
    lb = n_rows * k * (k + 1) // 2

    # --- Phase 1: Deterministic interleaved greedy (baseline) ---
    rows, used = _interleaved_greedy(n_rows, k, 200)
    if rows is not None:
        best_rows = rows
        best_scope = _scope(rows)

    if _dead():
        return best_rows

    # --- Phase 2: Try greedy fill at various scopes ---
    for m in range(max(lb, 100), min(lb + 100, 200)):
        if _dead():
            break
        rows, used = _greedy_fill(n_rows, k, m, rnd)
        if rows is not None and _valid(rows, n_rows, k):
            sc = _scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = rows
            break

    if _dead():
        return best_rows

    # --- Phase 3: Simulated annealing to improve ---
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 2.0
        if remaining > 5.0:
            improved, improved_scope = _simulated_annealing(
                best_rows, n_rows, k, remaining, rnd
            )
            if improved is not None and improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = improved

    if _dead():
        return best_rows

    # --- Phase 4: Randomized interleaved restarts ---
    for restart in range(30):
        if _dead():
            break
        max_mark = best_scope - 1 if best_scope < float('inf') else 150
        if max_mark < k:
            max_mark = best_scope

        rows = [[0] for _ in range(n_rows)]
        used = set()
        success = True

        for _ in range(n_rows * k):
            candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
            if not candidates:
                break
            # Weighted random choice favoring smaller max
            weights = [1.0 / (rows[i][-1] + 1) for i in candidates]
            total_w = sum(weights)
            r = rnd.random() * total_w
            cum = 0
            chosen = candidates[0]
            for idx, i in enumerate(candidates):
                cum += weights[idx]
                if r <= cum:
                    chosen = i
                    break
            row = rows[chosen]
            x = row[-1] + 1
            found = False
            while x <= max_mark:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
                x += 1
            if not found:
                success = False
                break

        if success and all(len(r) == k + 1 for r in rows):
            sc = _scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = rows

    if _dead():
        return best_rows

    # --- Phase 5: Local search – try to reduce individual entries ---
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 2.0
        if remaining > 1.0:
            best = [r[:] for r in best_rows]
            best_sc = _scope(best)
            used = set()
            for r in best:
                used |= _diffs_of_row(r)

            deadline_local = time.time() + remaining
            improved = True
            while improved and time.time() < deadline_local:
                improved = False
                for ri in range(n_rows):
                    row = best[ri]
                    for pi in range(1, len(row)):
                        old_val = row[pi]
                        for new_val in range(old_val - 1, 0, -1):
                            if pi > 0 and new_val <= row[pi - 1]:
                                continue
                            if pi < len(row) - 1 and new_val >= row[pi + 1]:
                                continue
                            if new_val <= 0:
                                continue
                            old_diffs = _diffs_of_row(row)
                            # Create modified row
                            new_row = row[:]
                            new_row[pi] = new_val
                            new_diffs = _diffs_of_row(new_row)
                            remaining_used = used - old_diffs
                            if not new_diffs & remaining_used:
                                used = remaining_used | new_diffs
                                row[pi] = new_val
                                new_sc = _scope(best)
                                if new_sc < best_sc:
                                    best_sc = new_sc
                                    improved = True
                                break

            if best_sc < best_scope:
                best_scope = best_sc
                best_rows = best

    return best_rows
