"""Difference triangle set (7,5) - greedy + local search + bounded backtracking."""
import time

_DEADLINE = 120.0  # safety margin under the 150s wall-clock limit


def _row_diffs(row):
    """All positive differences a_j - a_i for i < j in a single row."""
    d = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.append(row[j] - ai)
    return d


def _used_set(rows):
    used = set()
    for r in rows:
        used.update(_row_diffs(r))
    return used


def _scope(rows):
    return max(max(r) for r in rows)


def _greedy(n_rows, k, used=None, rows=None):
    """Greedy ruler construction. Returns (rows, used)."""
    if used is None:
        used = set()
    if rows is None:
        rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows, used


def _try_move_last(rows, used, k, deadline):
    """Try to reduce the last mark of each row. Returns (rows, used, improved)."""
    improved = False
    n = len(rows)
    for i in range(n):
        row = rows[i]
        old_last = row[-1]
        # differences involving the last element
        old_new = {old_last - a for a in row[:-1]}
        used -= old_new
        # try smaller values
        for v in range(old_last - 1, row[-2], -1):
            cand_new = {v - a for a in row[:-1]}
            if not (cand_new & used):
                row[-1] = v
                used |= cand_new
                improved = True
                break
        else:
            # no improvement; restore
            used |= old_new
        if time.time() > deadline:
            break
    return rows, used, improved


def _local_search(rows, used, k, deadline):
    """Repeatedly try to shrink rows. Returns (rows, used)."""
    while time.time() < deadline:
        rows, used, improved = _try_move_last(rows, used, k, deadline)
        if not improved:
            break
    return rows, used


def _backtrack(n_rows, k, used, rows, idx, deadline):
    """DFS backtracking for small problems. Returns (rows, used) or None."""
    if time.time() > deadline:
        return None
    if idx == n_rows:
        return rows, used
    row = [0]
    x = 0
    # build the row with backtracking
    def dfs(row, x, used, deadline):
        if time.time() > deadline:
            return None
        if len(row) == k + 1:
            # try next row
            res = _backtrack(n_rows, k, used, rows + [row], idx + 1, deadline)
            return res
        # try x, x+1, ...
        for v in range(x, x + 200):  # bounded search depth
            new = {v - a for a in row}
            if not (new & used):
                res = dfs(row + [v], v, used | new, deadline)
                if res is not None:
                    return res
            if time.time() > deadline:
                return None
        return None
    return dfs(row, x, used, deadline)


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE - 5.0  # safety margin

    # Phase 1: fast greedy baseline (always valid)
    rows, used = _greedy(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)

    # Phase 2: local search to shrink
    rows, used = _local_search(rows, used, k, deadline)
    if _scope(rows) < best_scope:
        best_rows = [list(r) for r in rows]
        best_scope = _scope(best_rows)

    # Phase 3: if small enough, try exact backtracking
    # Total differences needed: n_rows * k(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    # A rough size estimate: if scope would be < ~40, backtracking is feasible
    if best_scope < 45 and time.time() < deadline - 10.0:
        # restart from scratch with backtracking
        rows2, used2 = _backtrack(n_rows, k, set(), [], 0, deadline)
        if rows2 is not None:
            s2 = _scope(rows2)
            if s2 < best_scope:
                best_rows = [list(r) for r in rows2]
                best_scope = s2

    return best_rows
