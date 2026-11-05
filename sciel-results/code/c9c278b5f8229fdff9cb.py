"""Difference triangle set (4,7): minimize the largest entry."""
import time

# EVOLVE-BLOCK-START
def _all_diffs(rows):
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                used.add(row[j] - ai)
    return used


def _is_valid(rows):
    # all positive differences distinct globally; each row strictly increasing from 0
    used = set()
    for row in rows:
        if row[0] != 0:
            return False
        prev = -1
        for a in row:
            if a <= prev:
                return False
            prev = a
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                if d in used:
                    return False
                used.add(d)
    return True


def _greedy_construct(n_rows, k):
    """Each row: start at 0, pick smallest next mark whose diffs are all unused."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _try_lower_row(rows, r, used, deadline):
    """Try to reduce the largest entry of row r by lowering marks from the end.
    Returns True if any change was made."""
    row = rows[r]
    n = len(row)
    # Try to lower the maximum first, then each mark from the end inward.
    for idx in range(n - 1, 0, -1):
        orig = row[idx]
        # candidate values below orig, down to row[idx-1] + 1
        for cand in range(orig - 1, row[idx - 1], -1):
            # compute new differences contributed by this mark
            # remove old diffs involving idx, add new
            old = set()
            for j in range(idx):
                old.add(orig - row[j])
            new = set()
            ok = True
            for j in range(idx):
                d = cand - row[j]
                if d in used or d in new:
                    ok = False
                    break
                new.add(d)
            if not ok:
                continue
            # also must not collide with diffs of marks after idx (none involve idx as lower)
            # diffs with later marks: row[later] - cand must be unused
            for j in range(idx + 1, n):
                d = row[j] - cand
                if d in used or d in new:
                    ok = False
                    break
                new.add(d)
            if not ok:
                continue
            # valid: apply
            for d in old:
                used.discard(d)
            used |= new
            row[idx] = cand
            return True
    return False


def _local_improve(rows, n_rows, k, deadline):
    used = _all_diffs(rows)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r in range(n_rows):
            if time.time() >= deadline:
                break
            if _try_lower_row(rows, r, used, deadline):
                improved = True
    return rows


def construct(n_rows, k):
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s

    rows = _greedy_construct(n_rows, k)
    best = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]

    # Local improvement: try to shrink the scope.
    rows = _local_improve(rows, n_rows, k, deadline)
    cur = max(max(r) for r in rows)
    if cur < best:
        best = cur
        best_rows = [list(r) for r in rows]

    return best_rows
# EVOLVE-BLOCK-END
