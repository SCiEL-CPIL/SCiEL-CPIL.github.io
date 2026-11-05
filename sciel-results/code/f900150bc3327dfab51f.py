"""Difference triangle set: n_rows rows of k+1 increasing ints from 0, all
positive within-row differences distinct across all rows. Minimize max entry."""
import time
import random
import math

# EVOLVE-BLOCK-START

_DEADLINE = 115.0
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _is_valid(rows):
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                if d in seen:
                    return False
                seen.add(d)
    return True


def _score(rows):
    # lower is better: the largest entry
    return max(max(r) for r in rows)


def _row_differences(row):
    out = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            out.append(row[j] - ai)
    return out


def _construct_greedy(n_rows, k, rng=None):
    """Greedy: each row picks smallest next mark whose differences are unused."""
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


def _construct_greedy_random(n_rows, k, rng):
    """Greedy with occasional random skips to escape local structure."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # sometimes jump ahead to break determinism
            step = 1
            if rng.random() < 0.15:
                step = rng.randint(1, 4)
            for _s in range(step):
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
                    break
            else:
                # fallback: advance one by one
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
        rows.append(row)
    return rows


def _random_row(k, max_val, rng):
    """Random strictly increasing row starting at 0 within [0, max_val]."""
    pts = sorted(rng.sample(range(1, max_val + 1), k))
    return [0] + pts


def _try_repair(rows, n_rows, k, max_val, rng, max_iters=2000):
    """Local search: mutate rows (change an entry) to reduce max while keeping valid."""
    best = _score(rows)
    cur = [list(r) for r in rows]
    it = 0
    while it < max_iters and _elapsed() < _DEADLINE:
        it += 1
        # pick a random row and a random position (not 0, not last maybe) to perturb
        ri = rng.randrange(n_rows)
        row = cur[ri]
        pos = rng.randrange(1, len(row))
        old = row[pos]
        # try to move this entry to a nearby new value
        delta = rng.choice([-2, -1, 1, 2, 3, 5])
        newv = old + delta
        if newv <= row[pos - 1] or (pos + 1 < len(row) and newv >= row[pos + 1]):
            continue
        if newv > max_val:
            continue
        row[pos] = newv
        # validate whole thing quickly
        if _is_valid(cur):
            s = _score(cur)
            if s < best:
                best = s
        else:
            row[pos] = old
    return cur, best


def _local_search_improve(rows, n_rows, k, rng, budget=8.0):
    """Repeatedly mutate and keep improvements; target the max element."""
    best_rows = [list(r) for r in rows]
    best = _score(best_rows)
    max_val = best
    start = time.time()
    while time.time() - start < budget and _elapsed() < _DEADLINE:
        # attempt a targeted change on the row(s) holding the max
        cur = [list(r) for r in best_rows]
        # find entries equal to max
        candidates = []
        for ri, row in enumerate(cur):
            for pi, v in enumerate(row):
                if v == max_val and pi != 0:
                    candidates.append((ri, pi, v))
        if not candidates:
            break
        ri, pi, v = rng.choice(candidates)
        row = cur[ri]
        # try to lower v to a smaller unused-in-row value
        lower = row[pi - 1] + 1
        upper = (row[pi + 1] - 1) if pi + 1 < len(row) else max_val - 1
        if upper < lower:
            continue
        # sample a few candidate new values
        tried = 0
        improved = False
        while tried < 12 and not improved:
            tried += 1
            newv = rng.randint(lower, upper)
            if newv == v:
                continue
            row[pi] = newv
            if _is_valid(cur):
                s = _score(cur)
                if s < best:
                    best = s
                    best_rows = [list(r) for r in cur]
                    max_val = best
                    improved = True
                else:
                    row[pi] = v
            else:
                row[pi] = v
    return best_rows, best


def _backtrack_construct(n_rows, k, max_val, rng, node_budget=400000):
    """Bounded randomized backtracking: assign rows one by one, entries one by one,
    within [0, max_val], all differences distinct. Returns rows or None."""
    used = set()
    rows = []
    nodes = 0

    def rec(ri, row, x):
        nonlocal nodes
        nodes += 1
        if nodes > node_budget:
            return False
        if _elapsed() > _DEADLINE:
            return False
        if ri == n_rows:
            rows.append(row)
            return True
        # build next row
        return _build_row(ri, row, x)

    def _build_row(ri, row, x):
        nonlocal nodes
        if len(row) == k + 1:
            return rec(ri + 1, [], 0)
        # choose next value > last
        start = (row[-1] + 1) if row else 1
        # candidate values
        cands = list(range(start, max_val + 1))
        # order: prefer smaller, but shuffle a bit for randomness
        if rng.random() < 0.5:
            rng.shuffle(cands)
        else:
            cands.sort()
        for v in cands:
            if v > max_val:
                break
            new = {v - a for a in row}
            if new & used:
                continue
            nodes += 1
            if nodes > node_budget:
                return False
            used |= new
            row.append(v)
            if _build_row(ri, row, v):
                return True
            row.pop()
            used -= new
        return False

    ok = _build_row(0, [0], 0)
    if not ok:
        return None
    return rows


def construct(n_rows, k):
    global _DEADLINE, _T0
    _T0 = time.time()
    # safety margin before 150s hard kill
    _DEADLINE = 115.0

    # quick deterministic baseline
    base = _construct_greedy(n_rows, k)
    best_rows = base
    best = _score(best_rows)

    rng = random.Random(12345)

    # 1) randomized greedy variants
    for _ in range(60):
        if _elapsed() > _DEADLINE:
            break
        rows = _construct_greedy_random(n_rows, k, rng)
        if _is_valid(rows):
            s = _score(rows)
            if s < best:
                best = s
                best_rows = rows

    # 2) random rows + local search to reduce max
    # estimate a starting max_val from the counting bound / current best
    # counting bound: total diffs = n_rows * k(k+1)/2 must be <= max_val
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = int(math.ceil(math.sqrt(2 * total_diffs))) - 1
    # we'll try to search in [lower_bound, best]
    for _ in range(40):
        if _elapsed() > _DEADLINE:
            break
        max_val = best
        rows = []
        ok = True
        used = set()
        for _r in range(n_rows):
            row = None
            for _att in range(200):
                cand = _random_row(k, max_val, rng)
                if not (_row_differences(cand) and any(d in used for d in _row_differences(cand))):
                    row = cand
                    used |= set(_row_differences(cand))
                    break
            if row is None:
                ok = False
                break
            rows.append(row)
        if ok and _is_valid(rows):
            s = _score(rows)
            if s < best:
                best = s
                best_rows = rows

    # 3) local search attacking the max element
    if _elapsed() < _DEADLINE:
        improved_rows, improved = _local_search_improve(
            best_rows, n_rows, k, rng, budget=min(20.0, max(2.0, _DEADLINE - _elapsed() - 5.0))
        )
        if improved < best:
            best = improved
            best_rows = improved_rows

    # 4) bounded backtracking trying to beat current best, descending max_val
    if _elapsed() < _DEADLINE:
        # try max_val from best-1 down toward lower_bound, limited
        for mv in range(best - 1, max(lower_bound, 0), -1):
            if _elapsed() > _DEADLINE - 5.0:
                break
            rows = _backtrack_construct(n_rows, k, mv, rng, node_budget=150000)
            if rows is not None and _is_valid(rows):
                best = mv
                best_rows = rows
                break

    # final safety: ensure validity
    if not _is_valid(best_rows):
        # fall back to a guaranteed-valid greedy
        best_rows = _construct_greedy(n_rows, k)

    return best_rows

# EVOLVE-BLOCK-END
