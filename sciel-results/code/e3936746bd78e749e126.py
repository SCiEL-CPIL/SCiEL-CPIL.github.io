"""Difference triangle set (n_rows, k): minimize the largest entry.

Strategy (within the greedy + backtracking + bitset family):
  1. Fast deterministic greedy baseline (smallest next mark, unused differences).
  2. Decreasing-target backtracking: try to fit all rows within a decreasing
     scope T, using a bitset (set of ints) for O(1) difference-conflict checks.
  3. Local improvement + compaction polish on the best found solution.
An explicit deadline guard returns the best valid result found before the limit.
"""
import time

# Hard wall-clock limit; keep a safety margin below the 150 s evaluator kill.
_DEADLINE = 145.0
_t0 = time.time()


def _elapsed():
    return time.time() - _t0


def _diffs_of_row(row):
    """All positive differences a_j - a_i for i < j in row."""
    d = set()
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            d.add(row[j] - ai)
    return d


def _greedy(n_rows, k):
    """Deterministic greedy baseline: each row takes the smallest next mark whose
    differences to its earlier marks are all still unused by any row."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _backtrack(n_rows, k, scope, deadline):
    """Try to build all rows with every entry <= scope.
    Returns a list of rows on success, else None.

    Each row is a strictly increasing sequence of k+1 ints in [0, scope]
    starting at 0; differences are globally unique (bitset `used`).
    """
    used = set()
    rows = []

    def build_row(idx):
        row = [0]
        # choose k more marks in increasing order, each <= scope
        def rec(lo):
            if len(row) == k + 1:
                rows.append(list(row))
                return True
            if _elapsed() > deadline:
                return False
            # upper bound on next mark: must leave room for remaining marks
            remaining = (k + 1) - len(row)
            hi = scope - (remaining - 1)
            # also must exceed lo
            for x in range(lo + 1, hi + 1):
                new = {x - a for a in row}
                if new & used:
                    continue
                row.append(x)
                used |= new
                if rec(x):
                    return True
                used -= new
                row.pop()
            return False
        return rec(0)

    def rec_rows(ri):
        if ri == n_rows:
            return True
        if _elapsed() > deadline:
            return False
        if build_row(ri):
            if rec_rows(ri + 1):
                return True
            # undo: pop last row and its differences
            last = rows.pop()
            used -= _diffs_of_row(last)
        return False

    if rec_rows(0):
        return rows
    return None


def _polish(rows, k, deadline):
    """Local improvement + compaction: try to reduce the max entry.
    For each row, try to replace its last mark with a smaller valid value,
    then compact internal marks downward. Always returns a valid set."""
    used = set()
    for r in rows:
        used |= _diffs_of_row(r)

    improved = True
    while improved and _elapsed() < deadline:
        improved = False
        for ri, row in enumerate(rows):
            if len(row) != k + 1:
                continue
            # Try to reduce the last (largest) mark of this row.
            cur_last = row[-1]
            row_d = _diffs_of_row(row)
            used_wo = used - row_d
            # candidate smaller last marks: must keep increasing order
            # and all differences distinct from used_wo and internal ones.
            best = None
            for cand in range(row[-2] + 1, cur_last):
                trial = row[:-1] + [cand]
                td = _diffs_of_row(trial)
                if td & used_wo:
                    continue
                # also internal differences of row[:-1] must be consistent
                # (they are, since trial is increasing)
                best = cand
                break
            if best is not None:
                old_d = _diffs_of_row(row)
                new_d = _diffs_of_row(row[:-1] + [best])
                used -= old_d
                used |= new_d
                rows[ri] = row[:-1] + [best]
                improved = True
            # Compaction: try to shift internal marks (1..k-1) down by 1
            # if it stays increasing and differences stay distinct.
            for step in range(1, k):
                # attempt to reduce row[step] by 1
                if row[step] - 1 > row[step - 1] and row[step] - 1 < row[step + 1]:
                    trial = list(row)
                    trial[step] -= 1
                    td = _diffs_of_row(trial)
                    if not (td & (used - _diffs_of_row(row))):
                        used -= _diffs_of_row(row)
                        used |= td
                        rows[ri] = trial
                        improved = True
                        break
    return rows


def construct(n_rows, k):
    deadline = _DEADLINE - 3.0  # safety margin
    # 1. Fast baseline.
    rows = _greedy(n_rows, k)
    best_scope = max(max(r) for r in rows)

    # 2. Decreasing-target backtracking: try to beat the baseline scope.
    for T in range(best_scope - 1, 0, -1):
        if _elapsed() > deadline:
            break
        res = _backtrack(n_rows, k, T, deadline)
        if res is not None:
            rows = res
            best_scope = max(max(r) for r in rows)
            break

    # 3. Polish the best found.
    rows = _polish(rows, k, deadline)

    return rows
