import time
import random
import sys

_DEADLINE = 145.0  # safety margin under the 150s kill

def _row_candidates(row, used, k, M):
    """Yield (new_mark, new_diffs_bitset) for extending row, in increasing mark order.
    Only marks <= M are considered. new_diffs must not intersect used."""
    last = row[-1]
    for x in range(last + 1, M + 1):
        # differences x - a for a in row
        bad = False
        ds = 0
        for a in row:
            d = x - a
            if d & used:
                bad = True
                break
            ds |= (1 << d)
        if not bad:
            yield x, ds


def _greedy_construct(n_rows, k, M, rng):
    """Deterministic-ish greedy with small randomization. Returns (rows, max_entry) or None."""
    used = 0
    rows = []
    # build rows in a randomized order of "difficulty" is complex; just greedy
    order = list(range(n_rows))
    rng.shuffle(order)
    for r in order:
        row = [0]
        while len(row) < k + 1:
            found = False
            # try marks from row[-1]+1 upward
            for x in range(row[-1] + 1, M + 1):
                ds = 0
                ok = True
                for a in row:
                    d = x - a
                    if d & used:
                        ok = False
                        break
                    ds |= (1 << d)
                if ok:
                    row.append(x)
                    used |= ds
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    # reorder back to original index order
    rows.sort(key=lambda r: r[-1])  # not needed for validity; keep as-is
    return rows, max(r[-1] for r in rows)


def _backtrack(n_rows, k, M, used, rows, rng, deadline, counter):
    """
    rows: list of partial rows (length <= k+1); index i is next to complete.
    We always work on the "active" row = the one with fewest options (MRV).
    Returns (success, rows, used).
    """
    if time.time() > deadline:
        return False, rows, used
    counter[0] += 1
    if counter[0] % 20000 == 0 and time.time() > deadline:
        return False, rows, used

    # Find active row: among incomplete rows, the one with fewest valid next marks.
    best_idx = -1
    best_options = None
    best_count = None
    for i in range(n_rows):
        row = rows[i]
        if len(row) == k + 1:
            continue
        # count options (cap to avoid overwork)
        cnt = 0
        opts = []
        last = row[-1]
        for x in range(last + 1, M + 1):
            ds = 0
            ok = True
            for a in row:
                d = x - a
                if d & used:
                    ok = False
                    break
                    ds = 0
                    break
            if ok:
                ds = 0
                for a in row:
                    ds |= (1 << (x - a))
                opts.append((x, ds))
                cnt += 1
                if cnt >= 64:  # cap
                    break
        if cnt == 0:
            return False, rows, used
        if best_count is None or cnt < best_count:
            best_count = cnt
            best_idx = i
            best_options = opts
            if cnt == 1:
                break
    if best_idx == -1:
        # all complete
        return True, rows, used

    row = rows[best_idx]
    # global feasibility: remaining differences needed
    rem_diffs = 0
    for i in range(n_rows):
        L = len(rows[i])
        if L < k + 1:
            rem_diffs += (k + 1 - L) * (k + 1) // 2
    avail = bin((M & ~used) - 1).count("1") if M > 0 else 0
    # simpler: count available in [1, M]
    avail = 0
    u = used
    for d in range(1, M + 1):
        if not (u >> d) & 1:
            avail += 1
    if avail < rem_diffs:
        return False, rows, used

    # order options: smallest mark first, but with random tie-break
    rng.shuffle(best_options)
    best_options.sort(key=lambda t: t[0])
    for x, ds in best_options:
        rows2 = rows[:]
        rows2[best_idx] = row + [x]
        ok, rows2, used2 = _backtrack(n_rows, k, M, used | ds, rows2, rng, deadline, counter)
        if ok:
            return True, rows2, used2
    return False, rows, used


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)

    # total differences per row
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    # lower bound on M: need at least total_diffs distinct diffs in [1, M]
    # M(M+1)/2 >= total_diffs
    import math
    lb = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    lb = max(lb, k)  # also need k+1 marks

    best_rows = None
    best_scope = None

    # First try greedy for a quick valid answer
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        r = _greedy_construct(n_rows, k, 300, rng)
        if r is not None:
            rows, scope = r
            if best_scope is None or scope < best_scope:
                best_scope = scope
                best_rows = rows
            if scope <= lb:
                return best_rows

    # Now backtrack to try to beat the greedy scope
    if best_rows is None:
        # fallback: pure greedy without cap
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                ds = 0
                ok = True
                for a in row:
                    d = x - a
                    if d & used:
                        ok = False
                        break
                if ok:
                    ds = 0
                    for a in row:
                        ds |= (1 << (x - a))
                    row.append(x)
                    used |= ds
            rows.append(row)
        return rows

    # Try to improve best_scope via backtracking, decreasing M
    for M in range(min(best_scope, lb + 12), max(lb, 1) - 1, -1):
        if time.time() > deadline - 3:
            break
        if M < lb:
            break
        # randomized restarts
        found = False
        for restart in range(60):
            if time.time() > deadline - 2:
                break
            rng2 = random.Random(9999 + restart * 7 + M)
            counter = [0]
            rows_init = [[0] for _ in range(n_rows)]
            ok, rows_out, used_out = _backtrack(n_rows, k, M, 0, rows_init, rng2, deadline - 1, counter)
            if ok:
                scope = max(r[-1] for r in rows_out)
                if scope <= M and (best_scope is None or scope < best_scope):
                    best_scope = scope
                    best_rows = rows_out
                found = True
                break
        if found and best_scope == lb:
            break

    return best_rows
