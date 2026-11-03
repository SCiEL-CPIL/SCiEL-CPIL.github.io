"""Difference triangle set solver: backtracking CSP over difference values."""
import time
import sys

sys.setrecursionlimit(100000)


def _validate(rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _greedy(n_rows, k):
    """Fast greedy: smallest next mark with no difference collision."""
    used, rows = set(), []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            ok = True
            new = []
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
                new.append(d)
            if ok:
                row.append(x)
                used.update(new)
        rows.append(row)
    return rows


def _can_complete(row, used, remaining, max_mark):
    """Cheap feasibility: can we still add `remaining` marks in (row[-1], max_mark]
    without colliding with used? Greedy probe."""
    if remaining == 0:
        return True
    # probe by greedily placing
    probe = list(row)
    probe_used = set(used)
    x = probe[-1]
    placed = 0
    while len(probe) < len(row) + remaining and x < max_mark:
        x += 1
        ok = True
        new = []
        for a in probe:
            d = x - a
            if d in probe_used:
                ok = False
                break
            new.append(d)
        if ok:
            probe.append(x)
            probe_used.update(new)
            placed += 1
    return placed == remaining


def _backtrack(n_rows, k, deadline, best_holder):
    """Backtracking with forward checking. Returns list of rows or None."""
    total_diffs = n_rows * k * (k + 1) // 2  # C(k+1,2) per row
    # We need total_diffs distinct differences. The smallest possible max difference
    # lower bound: we need differences covering 1..m at least partially.
    used = set()
    rows = []
    # best max so far
    best_max = [best_holder[0]]

    # ordering: place rows; within a row place marks ascending
    def solve():
        if len(rows) == n_rows:
            cur_max = max(max(r) for r in rows)
            if cur_max < best_max[0]:
                best_max[0] = cur_max
                best_holder[1] = [list(r) for r in rows]
            return True
        if time.time() > deadline:
            return False
        # start a new row
        row = [0]
        _solve_row(row, used, rows, k, best_max)
        return False

    def _solve_row(row, used, rows, k, best_max):
        if time.time() > deadline:
            return False
        if len(row) == k + 1:
            # row complete; check if max improves potential
            rows.append(row)
            try:
                solve()
            finally:
                rows.pop()
            return False
        remaining = (k + 1) - len(row)
        x = row[-1]
        # try candidate marks
        while True:
            x += 1
            if time.time() > deadline:
                return False
            # compute new differences
            new = []
            ok = True
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
                new.append(d)
            if not ok:
                continue
            # forward check: can we complete the rest of this row?
            if remaining > 1:
                if not _can_complete(row, used, remaining - 1, x + 2 * remaining * k):
                    continue
            # commit
            row.append(x)
            for d in new:
                used.add(d)
            # recurse to next mark
            _solve_row(row, used, rows, k, best_max)
            # undo
            for d in new:
                used.discard(d)
            row.pop()
            # we keep trying larger x (this is the backtracking: we don't return
            # after first success so we can find a smaller max)
            # but to bound time, stop if we found a good solution already
            if best_holder[1] is not None and x > best_max[0]:
                break
            if time.time() > deadline:
                return False
        return False

    solve()
    return best_holder[1]


def construct(n_rows, k):
    start = time.time()
    deadline = start + 115.0  # safety margin under 120s / 150s

    # Always produce a valid greedy baseline immediately.
    greedy_rows = _greedy(n_rows, k)
    best = greedy_rows
    best_scope = _scope(best)

    # Try to improve via backtracking with a time budget.
    if time.time() < deadline - 5.0:
        best_holder = [best_scope, None]
        try:
            improved = _backtrack(n_rows, k, deadline, best_holder)
            if improved is not None and _validate(improved, k):
                if _scope(improved) < best_scope:
                    best = improved
                    best_scope = _scope(best)
        except RecursionError:
            pass
        except Exception:
            pass

    # Final safety: if no improvement, return greedy.
    if not _validate(best, k):
        # regenerate a guaranteed-valid greedy
        best = _greedy(n_rows, k)
    return best
