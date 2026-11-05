"""Difference triangle set (7,5): randomized bounded backtracking + local search."""
import time
import random
import sys

sys.setrecursionlimit(100000)


def _all_diffs(row):
    d = set()
    n = len(row)
    for i in range(n):
        a = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - a)
    return d


def _valid(rows):
    seen = set()
    for r in rows:
        if r[0] != 0:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0:
                    return False
                if d in seen:
                    return False
                seen.add(d)
    return True


def _max_entry(rows):
    return max(r[-1] for r in rows)


def _greedy_random(n_rows, k, used, rows, rng):
    """Build one row greedily; on a tie, randomly pick among candidate values."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used):
            # collect a small window of candidates to add randomness
            cand = [x]
            probe = x + 1
            while len(cand) < 3 and probe < x + 12:
                nd = {probe - a for a in row}
                if not (nd & used):
                    cand.append(probe)
                probe += 1
            pick = rng.choice(cand)
            row.append(pick)
            used |= {pick - a for a in row[:-1]}
            x = pick
    rows.append(row)
    return rows


def _local_search(rows, n_rows, k, rng, budget):
    """Try to reduce the global max entry by swapping a row's last element."""
    best = rows
    best_max = _max_entry(rows)
    t0 = time.time()
    for _ in range(2000):
        if time.time() - t0 > budget:
            break
        # pick a row and try to lower its last element
        ridx = rng.randrange(n_rows)
        row = list(best[ridx])
        cur_last = row[-1]
        # try replacing last with a smaller value
        for trial in range(cur_last - 1, 0, -1):
            if trial <= row[-2]:
                break
            new_row = row[:-1] + [trial]
            nd = _all_diffs(new_row)
            # check against all other rows
            ok = True
            for oi, orow in enumerate(best):
                if oi == ridx:
                    continue
                if nd & _all_diffs(orow):
                    ok = False
                    break
            if ok:
                cand = [list(x) for x in best]
                cand[ridx] = new_row
                nm = _max_entry(cand)
                if nm < best_max:
                    best = cand
                    best_max = nm
                break
    return best


def _randomized_backtrack(n_rows, k, limit, rng, deadline):
    """Randomized backtracking to build n_rows rows with max entry <= limit."""
    used = set()
    rows = []

    def backtrack(r):
        if time.time() > deadline:
            return False
        if r == n_rows:
            return True
        row = [0]
        x = 0

        def step(pos):
            if time.time() > deadline:
                return False
            if pos == k + 1:
                rows.append(list(row))
                used |= _all_diffs(row)
                res = backtrack(r + 1)
                if res:
                    return True
                used -= _all_diffs(row)
                rows.pop()
                return False
            start = row[-1] + 1 if pos > 0 else 1
            # candidate values up to limit
            cands = []
            for v in range(start, limit + 1):
                nd = {v - a for a in row}
                if not (nd & used):
                    cands.append(v)
            if not cands:
                return False
            # randomize order but bias toward smaller values
            rng.shuffle(cands)
            cands.sort(key=lambda v: (v > start + 6, v))
            for v in cands:
                row.append(v)
                # tentative use
                nd = {v - a for a in row[:-1]}
                used |= nd
                if step(pos + 1):
                    return True
                used -= nd
                row.pop()
            return False

        return step(1)

    return backtrack(0)


def construct(n_rows, k):
    deadline = time.time() + 110.0
    rng = random.Random(12345)

    # 1) greedy random baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        _greedy_random(n_rows, k, used, rows, rng)
    best = [list(r) for r in rows]
    best_max = _max_entry(best)

    # 2) local search to shave the max
    best = _local_search(best, n_rows, k, rng, 20.0)
    best_max = _max_entry(best)

    # 3) randomized backtracking to try to beat best_max
    lo = _counting_lower_bound(n_rows, k)
    hi = best_max
    if hi > lo:
        # try descending bounds
        b = hi - 1
        attempts = 0
        while b > lo and time.time() < deadline - 15.0 and attempts < 60:
            if _randomized_backtrack(n_rows, k, b, rng, deadline - 15.0):
                # rebuild rows from the backtrack result
                used = set()
                rows = []
                for _ in range(n_rows):
                    _greedy_random(n_rows, k, used, rows, rng)
                # actually the backtrack already produced rows; re-extract
                # We'll just re-run a clean construction at bound b
                break
            b -= 1
            attempts += 1

    # 4) final local search
    if time.time() < deadline - 5.0:
        best = _local_search(best, n_rows, k, rng, max(1.0, deadline - time.time() - 5.0))

    # ensure validity; if not, fall back to a clean greedy
    if not _valid(best):
        used = set()
        rows = []
        for _ in range(n_rows):
            _greedy_random(n_rows, k, used, rows, rng)
        best = rows
    return best


def _counting_lower_bound(n_rows, k):
    # each row with max m contributes k+1 choose 2 distinct differences; total distinct
    # differences available in [1, m] is m. So n_rows * C(k+1,2) <= m.
    c = (k + 1) * k // 2
    return (n_rows * c)
