"""Difference triangle set (4,7) - local search / simulated annealing."""
import time
import random
import sys
import math

_DEADLINE = 120.0          # hard wall-clock limit
_SAFETY = 8.0              # safety margin in seconds
_TIME_LIMIT = _DEADLINE - _SAFETY


def _row_diffs(row):
    """All positive differences within a single row."""
    d = []
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d.append(row[j] - ri)
    return d


def _used_set(rows):
    used = set()
    for r in rows:
        used.update(_row_diffs(r))
    return used


def _is_valid(rows):
    used = set()
    for r in rows:
        for d in _row_diffs(r):
            if d in used:
                return False
            used.add(d)
    return True


def _score(rows):
    """Objective: largest entry (lower better)."""
    return max(max(r) for r in rows)


def _greedy_row(used, k, start_bias=0):
    """Build one row greedily given already-used differences."""
    row = [0]
    x = start_bias
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    return row


def _greedy_init(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(_greedy_row(used, k))
    return rows


def _try_move(rows, used, move, k):
    """Apply a single move; return (new_used, ok) or None if invalid."""
    # move = (row_idx, pos_idx, new_val)  with pos_idx in 1..k
    ri, pi, nv = move
    row = rows[ri]
    old = row[pi]
    if nv <= row[pi - 1] or (pi + 1 < len(row) and nv >= row[pi + 1]):
        return None
    # differences affected: those involving position pi
    row_set = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            row_set.add(row[j] - row[i])
    # remove old diffs involving pi, add new
    affected = set()
    for i in range(len(row)):
        if i != pi:
            d = abs(row[pi] - row[i])
            affected.add(d)
    # rebuild row with new value
    row[pi] = nv
    new_row_set = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            new_row_set.add(row[j] - row[i])
    # net change in used
    removed = (row_set - new_row_set)
    added = (new_row_set - row_set)
    # check collisions
    ok = True
    for d in added:
        if d in used and d not in removed:
            ok = False
            break
    if not ok:
        row[pi] = old
        return None
    new_used = set(used)
    for d in removed:
        new_used.discard(d)
    for d in added:
        new_used.add(d)
    return new_used, True


def _local_search(rows, k, n_rows, deadline):
    """Simulated-annealing style local search."""
    used = _used_set(rows)
    best = _score(rows)
    best_rows = [list(r) for r in rows]
    best_used = set(used)

    # Precompute per-row diff sets for fast updates
    row_diffs = [set(_row_diffs(r)) for r in rows]

    T = 10.0
    T_min = 0.01
    cool = 0.9999

    rng = random.Random(12345)

    while time.time() < deadline:
        # pick a random row and position
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, len(rows[ri]))
        row = rows[ri]
        lo = row[pi - 1] + 1
        hi = row[pi + 1] - 1 if pi + 1 < len(row) else max(row) + 20
        if hi < lo:
            hi = lo
        nv = rng.randrange(lo, hi + 1)
        if nv == row[pi]:
            continue

        # compute net change
        old = row[pi]
        affected_old = set()
        for i in range(len(row)):
            if i != pi:
                affected_old.add(abs(old - row[i]))
        affected_new = set()
        for i in range(len(row)):
            if i != pi:
                affected_new.add(abs(nv - row[i]))
        removed = affected_old - affected_new
        added = affected_new - affected_old

        # check collisions with global used (excluding this row's own)
        ok = True
        for d in added:
            if d in used and d not in removed:
                ok = False
                break
        if not ok:
            continue

        # tentative objective: track max entry change
        old_max = max(row)
        new_max = max(nv, row[-1])
        old_score = max(max(r) for r in rows)
        # approximate: recompute only if this row had the max
        if old == old_max:
            new_row_score = new_max
            others_max = max(max(r) for r in rows if r is not row)
            new_score = max(new_row_score, others_max)
        else:
            new_score = old_score

        delta = new_score - old_score
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            # accept
            row[pi] = nv
            # update used
            for d in removed:
                used.discard(d)
            for d in added:
                used.add(d)
            # update row_diffs
            row_diffs[ri] = set()
            rr = row
            for i in range(len(rr)):
                for j in range(i + 1, len(rr)):
                    row_diffs[ri].add(rr[j] - rr[i])

            if new_score < best:
                best = new_score
                best_rows = [list(r) for r in rows]
                best_used = set(used)

        T *= cool
        if T < T_min:
            T = T_min

    return best_rows, best


def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT

    # Greedy initialization
    rows = _greedy_init(n_rows, k)
    best_rows = [list(r) for r in rows]
    best = _score(best_rows)

    # Local search
    if time.time() < deadline - 5:
        ls_rows, ls_best = _local_search([list(r) for r in rows], k, n_rows, deadline)
        if ls_best < best:
            best = ls_best
            best_rows = ls_rows

    # Final validity check
    if not _is_valid(best_rows):
        # fallback to greedy
        best_rows = _greedy_init(n_rows, k)

    return best_rows
