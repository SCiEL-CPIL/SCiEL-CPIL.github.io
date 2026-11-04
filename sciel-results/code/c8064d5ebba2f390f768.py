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


def _used_set(rows):
    used = set()
    for r in rows:
        used.update(_row_diffs(r))
    return used


def _scope(rows):
    return max(max(r) for r in rows)


def _greedy(n_rows, k, used=None, rows=None, start_offset=0):
    """Greedy ruler construction. Returns (rows, used)."""
    if used is None:
        used = set()
    if rows is None:
        rows = []
    for r_idx in range(n_rows):
        row, x = [0], start_offset
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows, used


def _try_move_element(rows, used, k, deadline, row_idx, elem_idx):
    """Try to reduce a specific element. Returns (improved, new_val or None)."""
    row = rows[row_idx]
    old_val = row[elem_idx]
    
    # Compute differences that involve this element
    # For element at position p with value v:
    #   diffs with earlier elements: v - row[i] for i < p
    #   diffs with later elements: row[j] - v for j > p
    earlier = row[:elem_idx]
    later = row[elem_idx+1:]
    
    old_diffs = set()
    for a in earlier:
        old_diffs.add(old_val - a)
    for b in later:
        old_diffs.add(b - old_val)
    
    # Determine valid range for new value
    low = (earlier[-1] + 1) if earlier else 1
    high = (later[0] - 1) if later else old_val
    
    if low > high:
        return False, None
    
    # Try to find a smaller valid value
    # First try values from old_val-1 down to low
    for v in range(old_val - 1, low - 1, -1):
        cand_diffs = set()
        for a in earlier:
            cand_diffs.add(v - a)
        for b in later:
            cand_diffs.add(b - v)
        
        if not (cand_diffs & used):
            return True, v
    
    return False, None


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
    """Repeatedly try to shrink rows by moving any element. Returns (rows, used)."""
    while time.time() < deadline:
        any_improved = False
        # Try moving elements from the end backwards, and from later positions first
        # This tends to find reductions that enable further reductions
        for i in range(len(rows)):
            row = rows[i]
            # Try elements from position k down to 1 (skip position 0 which is always 0)
            for p in range(len(row) - 1, 0, -1):
                improved, new_val = _try_move_element(rows, used, k, deadline, i, p)
                if improved and new_val is not None:
                    old_val = row[p]
                    earlier = row[:p]
                    later = row[p+1:]
                    
                    old_diffs = set()
                    for a in earlier:
                        old_diffs.add(old_val - a)
                    for b in later:
                        old_diffs.add(b - old_val)
                    
                    new_diffs = set()
                    for a in earlier:
                        new_diffs.add(new_val - a)
                    for b in later:
                        new_diffs.add(b - new_val)
                    
                    used -= old_diffs
                    used |= new_diffs
                    row[p] = new_val
                    any_improved = True
                    break
            if time.time() > deadline:
                break
        if not any_improved:
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

    # Phase 3: Multiple restarts with different greedy seeds + local search
    # Use randomized starting offsets to explore different greedy solutions
    rng = random.Random(42)
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        # Try different greedy constructions with varying start offsets
        offset = rng.randint(0, 30)
        rows_r, used_r = _greedy(n_rows, k, used=set(), rows=[], start_offset=offset)
        # Apply local search with a sub-deadline
        sub_deadline = min(deadline - 5.0, time.time() + 5.0)
        rows_r, used_r = _local_search(rows_r, used_r, k, sub_deadline)
        scope_r = _scope(rows_r)
        if scope_r < best_scope:
            best_rows = [list(r) for r in rows_r]
            best_scope = scope_r
            # If we found a significantly better solution, do more local search
            if time.time() < deadline - 10.0:
                rows_r, used_r = _local_search(rows_r, used_r, k, deadline - 5.0)
                scope_r2 = _scope(rows_r)
                if scope_r2 < best_scope:
                    best_rows = [list(r) for r in rows_r]
                    best_scope = scope_r2

    # Phase 4: if small enough, try exact backtracking
    if best_scope < 45 and time.time() < deadline - 10.0:
        rows2, used2 = _backtrack(n_rows, k, set(), [], 0, deadline)
        if rows2 is not None:
            s2 = _scope(rows2)
            if s2 < best_scope:
                best_rows = [list(r) for r in rows2]
                best_scope = s2

    return best_rows
