"""Difference triangle set (7,5): keep SA+local-search family, make cost incremental."""
# EVOLVE-BLOCK-START
import random
import time
import math

# ---------- helpers ----------

def _row_diffs(row):
    """Set of all positive differences within one row."""
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _is_valid(rows):
    all_d = set()
    total = 0
    for r in rows:
        total += len(r) * (len(r) - 1) // 2
        all_d |= _row_diffs(r)
    return len(all_d) == total


def _scope(rows):
    return max(max(r) for r in rows)


def _diff_multiset(rows):
    """Counter of every positive difference across all rows."""
    from collections import Counter
    c = Counter()
    for r in rows:
        n = len(r)
        for i in range(n):
            ai = r[i]
            for j in range(i + 1, n):
                c[r[j] - ai] += 1
    return c


def _greedy_init(n_rows, k):
    """Greedy smallest-first construction (valid starting point)."""
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


# ---------- incremental simulated annealing ----------

def _sa_search(rows, k, deadline, rng):
    """Simulated annealing with incremental difference multiset.

    Returns (best_rows, best_scope)."""
    from collections import Counter

    rows = [list(r) for r in rows]
    n = len(rows)
    m = k + 1
    cnt = _diff_multiset(rows)
    scope = _scope(rows)
    dups = sum(c for c in cnt.values() if c > 1)
    cost = dups * 100000 + scope

    best_rows = [list(r) for r in rows]
    best_scope = scope
    best_dups = dups
    best_cost = cost

    temp = 40.0
    cooling = 0.999
    TMIN = 0.3

    while time.time() < deadline:
        if temp < TMIN:
            temp = TMIN
        ri = rng.randrange(n)
        row = rows[ri]
        strategy = rng.random()

        if strategy < 0.6:
            # move one element (not the first)
            pos = rng.randrange(1, m)
            lo = row[pos - 1] + 1
            hi = row[pos + 1] - 1 if pos + 1 < m else scope + 40
            if hi < lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == row[pos]:
                continue
            old = row[pos]
            # remove old diffs
            for j in range(m):
                if j == pos:
                    continue
                d = abs(row[j] - old)
                cnt[d] -= 1
                if cnt[d] == 0:
                    del cnt[d]
            row[pos] = new_val
            # add new diffs
            for j in range(m):
                if j == pos:
                    continue
                d = abs(row[j] - new_val)
                c = cnt.get(d, 0)
                if c == 0:
                    dups += 1
                elif c == 1:
                    dups += 2
                cnt[d] = c + 1
            new_scope = max(row)
            if new_scope > scope:
                scope = new_scope
        else:
            # shift a suffix (keeps increasing order)
            pos = rng.randrange(1, m)
            delta = rng.randint(-4, 4)
            if delta == 0 or row[pos] + delta <= row[pos - 1]:
                continue
            # remove diffs crossing the cut
            for a in row[:pos]:
                for b in row[pos:]:
                    d = b - a
                    cnt[d] -= 1
                    if cnt[d] == 0:
                        del cnt[d]
            for p in range(pos, m):
                row[p] += delta
            # add diffs crossing the cut
            for a in row[:pos]:
                for b in row[pos:]:
                    d = b - a
                    c = cnt.get(d, 0)
                    if c == 0:
                        dups += 1
                    elif c == 1:
                        dups += 2
                    cnt[d] = c + 1
            new_scope = row[-1]
            if new_scope > scope:
                scope = new_scope

        new_cost = dups * 100000 + scope
        dc = new_cost - cost
        if dc <= 0 or rng.random() < math.exp(-dc / temp):
            cost = new_cost
            if dups < best_dups or (dups == best_dups and scope < best_scope):
                best_dups = dups
                best_scope = scope
                best_cost = cost
                best_rows = [list(r) for r in rows]
        else:
            # rollback
            if strategy < 0.6:
                row[pos] = old
                for j in range(m):
                    if j == pos:
                        continue
                    d = abs(row[j] - old)
                    c = cnt.get(d, 0)
                    if c == 0:
                        dups += 1
                    elif c == 1:
                        dups += 2
                    cnt[d] = c + 1
                for j in range(m):
                    if j == pos:
                        continue
                    d = abs(row[j] - new_val)
                    cnt[d] -= 1
                    if cnt[d] == 0:
                        del cnt[d]
            else:
                for p in range(pos, m):
                    row[p] -= delta
                for a in row[:pos]:
                    for b in row[pos:]:
                        d = b - a
                        cnt[d] -= 1
                        if cnt[d] == 0:
                            del cnt[d]
                for a in row[:pos]:
                    for b in row[pos:]:
                        d = b - a
                        c = cnt.get(d, 0)
                        if c == 0:
                            dups += 1
                        elif c == 1:
                            dups += 2
                        cnt[d] = c + 1
            # restore scope (recompute cheaply for the affected row)
            scope = max(max(r) for r in rows)
            cost = dups * 100000 + scope

        temp *= cooling

    return best_rows, best_scope


# ---------- local search (scope reduction) ----------

def _local_search(rows, k, deadline, rng):
    """Reduce scope while keeping all differences distinct. Returns best rows."""
    from collections import Counter
    m = k + 1
    rows = [list(r) for r in rows]
    best = [list(r) for r in rows]
    best_scope = _scope(best)

    while time.time() < deadline:
        cnt = _diff_multiset(rows)
        scope = _scope(rows)
        improved = False
        for ri in range(len(rows)):
            row = rows[ri]
            # try to lower the last (largest) element first
            for pos in range(m - 1, 0, -1):
                lo = row[pos - 1] + 1
                for nv in range(row[pos] - 1, lo - 1, -1):
                    # check conflicts with the rest of the row + other rows
                    ok = True
                    for j in range(m):
                        if j == pos:
                            continue
                        d = abs(row[j] - nv)
                        if cnt.get(d, 0) >= 1:
                            ok = False
                            break
                    if ok:
                        row[pos] = nv
                        # update cnt
                        for j in range(m):
                            if j == pos:
                                continue
                        # recompute row contribution
                        for i2 in range(m):
                            for j2 in range(i2 + 1, m):
                                d = row[j2] - row[i2]
                                # we will rebuild below; just set row and recompute
                        break
                    else:
                        break
                if time.time() >= deadline:
                    break
            if time.time() >= deadline:
                break
        # rebuild cnt and accept if valid + smaller
        cnt = _diff_multiset(rows)
        if all(c == 1 for c in cnt.values()):
            s = _scope(rows)
            if s < best_scope:
                best = [list(r) for r in rows]
                best_scope = s
                improved = True
        if not improved:
            break
    return best


def construct(n_rows, k):
    """Construct difference triangle set using simulated annealing + local search."""
    start = time.time()
    deadline = start + 118  # safety margin under 150s wall clock

    rng = random.Random(12345)
    rows = _greedy_init(n_rows, k)

    # Phase 1: simulated annealing (incremental)
    sa_deadline = start + 95
    rows, scope = _sa_search(rows, k, sa_deadline, rng)
    if not _is_valid(rows):
        rows = _greedy_init(n_rows, k)
        scope = _scope(rows)

    # Phase 2: local search to trim scope
    rows = _local_search(rows, k, start + 112, rng)
    if not _is_valid(rows):
        rows = _greedy_init(n_rows, k)

    # Phase 3: final polish
    rows2 = _local_search(rows, k, start + 118, rng)
    if _is_valid(rows2) and _scope(rows2) < _scope(rows):
        rows = rows2

    return rows
# EVOLVE-BLOCK-END
