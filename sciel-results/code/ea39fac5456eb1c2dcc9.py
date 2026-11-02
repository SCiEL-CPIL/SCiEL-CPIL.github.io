"""Difference triangle set (7,5): minimize the largest entry.

Design (within the ILS family that previously improved the island):
  1. Fast greedy baseline (smallest feasible mark per row).
  2. Per-row optimization: given the other rows' used differences, find a
     row (starting at 0, k+1 increasing ints) minimizing its largest entry
     via backtracking with pruning (the "local search" step).
  3. Iterated Local Search: randomly pick a row, re-optimize it against the
     current used set; accept if the global max drops. Repeat until deadline.
  4. Deadline guard: return the best valid result before the wall-clock limit.
"""
import time

_DEADLINE = 145.0  # safety margin under the 150 s kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _row_diffs(row):
    return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}


def _greedy_row(used, k, limit):
    """Smallest-feasible-mark row of length k+1 starting at 0."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if x > limit:
            break
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
    return row if len(row) == k + 1 else None


def _optimize_row(used, k, limit, deadline):
    """Backtrack for a row minimizing its last (largest) entry.

    Returns the best row found (may be None if nothing fits within limit),
    stopping if the deadline is reached.
    """
    best = [None]
    best_val = [limit + 1]

    def rec(row):
        if _elapsed() > deadline:
            return
        if len(row) == k + 1:
            v = row[-1]
            if v < best_val[0]:
                best_val[0] = v
                best[0] = row[:]
            return
        # lower bound: need k+1-len(row) more marks, each at least +1
        if row[-1] + (k + 1 - len(row)) >= best_val[0]:
            return
        x = row[-1]
        while x < best_val[0] and x <= limit:
            x += 1
            new = {x - a for a in row}
            if new & used:
                continue
            row.append(x)
            rec(row)
            row.pop()
            if best_val[0] <= x:  # can't beat with larger x
                break

    rec([0])
    return best[0]


def construct(n_rows, k):
    deadline = _START + _DEADLINE
    # ---- 1. greedy baseline ----
    used = set()
    rows = []
    for _ in range(n_rows):
        r = _greedy_row(used, k, 10 ** 9)
        if r is None:
            break
        rows.append(r)
        used |= _row_diffs(r)
    if len(rows) < n_rows:
        return rows
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)
    limit = best_max

    # ---- 2/3. ILS: re-optimize random rows against current used set ----
    import random
    rng = random.Random(12345)
    while _elapsed() < deadline - 1.0:
        if len(rows) < n_rows:
            break
        i = rng.randrange(n_rows)
        # used set excluding row i
        used_others = set()
        for j, r in enumerate(rows):
            if j != i:
                used_others |= _row_diffs(r)
        # try to beat current row i's max
        cur_max = max(rows[i])
        new_row = _optimize_row(used_others, k, cur_max - 1, deadline)
        if new_row is None:
            # also allow equal-length but maybe smaller; try up to cur_max
            new_row = _optimize_row(used_others, k, cur_max, deadline)
        if new_row is not None and max(new_row) < cur_max:
            rows[i] = new_row
            new_max = max(max(r) for r in rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in rows]
                limit = best_max
    return best_rows
