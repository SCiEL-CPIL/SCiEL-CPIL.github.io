"""Difference triangle set (4,7): minimize the largest entry (scope).

Design (greedy + randomized local improvement, bitmask difference tracking):
  * Each row is k+1 strictly increasing integers starting at 0.
  * All positive intra-row differences must be distinct across all rows.
  * Objective: minimize the maximum entry (the scope).

Speed / structure:
  * Differences are tracked with a uint64 bitmask (max diff is far below 2^64),
    so collision tests and un-adding a whole row are O(1) bitwise ops.
  * A fast greedy pass builds a valid starting solution.
  * Local search then re-derives one random row at a time (removing its
    differences from the global cloud, then rebuilding greedily with a small
    randomised lookahead) and keeps the change only if the scope shrinks.
  * An explicit wall-clock deadline returns the best valid solution found.
"""

import time
import random


def construct(n_rows, k):
    start = time.time()
    deadline = start + 110.0  # safety margin under the 150s kill
    random.seed(20250407)

    n = k + 1                       # entries per row
    # Number of differences per row (choose(k+1, 2)); total must fit in a bitmask.
    total_diffs = n_rows * (n * (n - 1) // 2)
    if total_diffs >= (1 << 63):
        # Fallback: extremely large instance, use a set-based greedy (rare).
        return _greedy_set(n_rows, k)

    def row_diff_mask(row):
        m = 0
        L = len(row)
        for i in range(L):
            ai = row[i]
            for j in range(i + 1, L):
                m |= 1 << (row[j] - ai)
        return m

    # ---------- Greedy construction (fast, exact) ----------
    def greedy_build(used, max_iter=400000):
        rows = []
        for _ in range(n_rows):
            row = [0]
            row_mask = 0
            x = 0
            while len(row) < n:
                x += 1
                # differences to earlier marks in this row
                new = 0
                for a in row:
                    new |= 1 << (x - a)
                if (new & used) == 0 and (new & row_mask) == 0:
                    row.append(x)
                    row_mask |= new
                    used |= new
                if x > max_iter:
                    return None  # failed (should not happen for small instances)
            rows.append(row)
        return rows

    # ---------- Local improvement ----------
    def local_improve(rows, lookahead, max_iters):
        best_rows = [r[:] for r in rows]
        best_scope = max(max(r) for r in best_rows)
        used = 0
        for r in best_rows:
            used |= row_diff_mask(r)
        it = 0
        while True:
            if time.time() > deadline:
                break
            it += 1
            if it > max_iters:
                break
            ri = random.randrange(n_rows)
            old = best_rows[ri]
            old_mask = row_diff_mask(old)
            # temporarily remove this row's differences from the global cloud
            used_without = used & ~old_mask

            # rebuild this row greedily with a small randomised lookahead
            row = [0]
            row_mask = 0
            x = 0
            ok = True
            while len(row) < n:
                x += 1
                cands = []
                for _ in range(lookahead):
                    cand_new = 0
                    for a in row:
                        cand_new |= 1 << (x - a)
                    if (cand_new & used_without) == 0 and (cand_new & row_mask) == 0:
                        cands.append((x, cand_new))
                        break
                    x += 1
                if not cands:
                    ok = False
                    break
                # among the first `lookahead` collision-free candidates, pick one
                # (prefer smaller, but allow a random jump to escape local minima)
                # collect up to `lookahead` candidates starting from x
                pool = []
                xx = x
                for _ in range(lookahead):
                    cn = 0
                    for a in row:
                        cn |= 1 << (xx - a)
                    if (cn & used_without) == 0 and (cn & row_mask) == 0:
                        pool.append((xx, cn))
                    xx += 1
                if not pool:
                    ok = False
                    break
                x, new = pool[random.randrange(len(pool))]
                row.append(x)
                row_mask |= new
                used_without |= new

            if not ok:
                continue
            new_scope = row[-1]
            if new_scope < best_scope:
                used = (used & ~old_mask) | row_mask
                best_rows[ri] = row
                best_scope = new_scope
        return best_rows, best_scope

    # ---------- Run ----------
    rows = greedy_build(0)
    if rows is None:
        # extremely unlikely; fall back to set-based greedy
        return _greedy_set(n_rows, k)

    best_scope = max(max(r) for r in rows)
    # A few passes with growing lookahead to escape local minima.
    for la in (1, 2, 3, 5):
        if time.time() > deadline - 5.0:
            break
        rows, best_scope = local_improve(rows, la, 200000)

    return rows


def _greedy_set(n_rows, k):
    """Set-based fallback greedy (used only if bitmask is infeasible)."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows
