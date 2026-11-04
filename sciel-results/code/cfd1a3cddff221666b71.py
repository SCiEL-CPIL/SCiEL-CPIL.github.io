"""Difference triangle set (7,5) - greedy + local search + bounded backtracking."""
import time
import random

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


def _all_diffs(rows):
    """All differences across all rows as a flat list."""
    diffs = []
    for r in rows:
        diffs.extend(_row_diffs(r))
    return diffs


def _used_set(rows):
    used = set()
    for r in rows:
        used.update(_row_diffs(r))
    return used


def _scope(rows):
    return max(max(r) for r in rows)


def _greedy(n_rows, k, used=None, rows=None, rng=None):
    """Greedy ruler construction. Returns (rows, used)."""
    if used is None:
        used = set()
    if rows is None:
        rows = []
    if rng is None:
        rng = random.Random(42)
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            # Try a window of candidates, pick the smallest that works
            # Occasionally skip ahead for diversity
            found = False
            for offset in range(0, 5):
                v = x + 1 + offset
                new = {v - a for a in row}
                if not (new & used):
                    row.append(v)
                    used |= new
                    x = v
                    found = True
                    break
            if not found:
                # Just increment until we find one
                x += 1
                while True:
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                        break
                    x += 1
        rows.append(row)
    return rows, used


def _greedy_randomized(n_rows, k, rng):
    """Randomized greedy: sometimes skip available positions for diversity."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            # Randomly decide how far to look
            skip = rng.randint(0, 3)
            found = False
            for offset in range(0, skip + 4):
                v = x + 1 + offset
                new = {v - a for a in row}
                if not (new & used):
                    row.append(v)
                    used |= new
                    x = v
                    found = True
                    break
            if not found:
                x += 1
                while True:
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                        break
                    x += 1
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


def _try_move_any(rows, used, k, deadline):
    """Try to reduce any non-first element of each row. Returns (rows, used, improved)."""
    improved = False
    n = len(rows)
    # Process rows in random-ish order (start from end for variety)
    order = list(range(n))
    for i in order:
        if time.time() > deadline:
            break
        row = rows[i]
        # Try to move each element (except first=0) to a smaller value
        for pos in range(len(row) - 1, 0, -1):  # from last to second
            old_val = row[pos]
            # Compute all diffs involving this element
            old_diffs = set()
            for j in range(pos):
                old_diffs.add(old_val - row[j])
            for j in range(pos + 1, len(row)):
                old_diffs.add(row[j] - old_val)
            
            # Remove these from used
            used -= old_diffs
            
            # Try smaller values
            lower_bound = row[pos - 1] + 1 if pos > 0 else 1
            found = False
            for v in range(old_val - 1, lower_bound - 1, -1):
                cand_diffs = set()
                for j in range(pos):
                    cand_diffs.add(v - row[j])
                for j in range(pos + 1, len(row)):
                    cand_diffs.add(row[j] - v)
                if not (cand_diffs & used):
                    row[pos] = v
                    used |= cand_diffs
                    improved = True
                    found = True
                    break
            if not found:
                used |= old_diffs
            if time.time() > deadline:
                return rows, used, improved
    return rows, used, improved


def _local_search(rows, used, k, deadline):
    """Repeatedly try to shrink rows using both last-element and any-element moves.
    Returns (rows, used)."""
    while time.time() < deadline:
        improved = False
        # First pass: move last elements
        rows, used, imp1 = _try_move_last(rows, used, k, deadline)
        improved |= imp1
        if time.time() > deadline:
            break
        # Second pass: move any elements
        rows, used, imp2 = _try_move_any(rows, used, k, deadline)
        improved |= imp2
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

    # Phase 3: randomized restarts with local search
    rng = random.Random(12345)
    remaining = deadline - time.time()
    if remaining > 10.0:
        # Allocate time for multiple restarts
        n_restarts = min(20, int(remaining // 3.0))
        for restart in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            restart_deadline = deadline - 5.0
            r_rows, r_used = _greedy_randomized(n_rows, k, rng)
            r_scope = _scope(r_rows)
            # Only do local search if promising
            if r_scope <= best_scope:
                r_rows, r_used = _local_search(r_rows, r_used, k, restart_deadline)
                r_scope = _scope(r_rows)
            if r_scope < best_scope:
                best_rows = [list(r) for r in r_rows]
                best_scope = r_scope

    # Phase 4: if small enough, try exact backtracking
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
