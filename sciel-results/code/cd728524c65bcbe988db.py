"""Difference triangle set (7,5): simultaneous MRV backtracking search."""
import time
import random

_DEADLINE = 150.0
_DEADLINE_MARGIN = 8.0
_T0 = time.monotonic()


def construct(n_rows, k):
    """Return list of n_rows rows, each k+1 increasing ints starting at 0,
    with all within-row positive differences distinct across rows.
    Minimizes the largest entry."""
    total = n_rows * k
    if total == 0:
        return [[] for _ in range(n_rows)]

    # Known good seed: the greedy construction (score 1.0 reference).
    greedy = _greedy(n_rows, k)
    best = greedy
    best_scope = max(max(r) for r in best)

    # Multi-start randomized greedy to find a tight upper bound for backtracking.
    rng = random.Random(12345)
    for _ in range(60):
        if time.monotonic() - _T0 > _DEADLINE - _DEADLINE_MARGIN:
            break
        cand = _rand_greedy(n_rows, k, rng)
        sc = max(max(r) for r in cand)
        if sc < best_scope:
            best_scope = sc
            best = cand

    # Backtrack search with MRV + global difference propagation.
    deadline = _DEADLINE - _DEADLINE_MARGIN
    found = _backtrack(n_rows, k, best_scope - 1, deadline, rng)
    if found is not None:
        sc = max(max(r) for r in found)
        if sc < best_scope:
            best_scope = sc
            best = found

    return best


def _greedy(n_rows, k):
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


def _rand_greedy(n_rows, k, rng):
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


def _backtrack(n_rows, k, scope, deadline, rng):
    """Search for a valid set with all entries <= scope. Returns list of rows or None."""
    # state: rows[i] = list of chosen values (starts [0]); next_i = index to fill.
    rows = [[0] for _ in range(n_rows)]
    used = set()
    # count of remaining elements per row
    remaining = [k] * n_rows

    def mrv_index():
        # choose row with fewest feasible candidates at its next slot
        best_i, best_count = -1, 10**9
        for i in range(n_rows):
            if remaining[i] == 0:
                continue
            last = rows[i][-1]
            # candidate must be > last and <= scope
            # count feasible: we approximate by scanning a window
            cnt = 0
            for v in range(last + 1, scope + 1):
                if not any((v - a) in used for a in rows[i]):
                    cnt += 1
                    if cnt >= best_count:
                        break
            if cnt < best_count:
                best_count = cnt
                best_i = i
                if cnt == 0:
                    return i, 0
        return best_i, best_count

    def solve(depth):
        if depth == n_rows * k:
            return True
        if time.monotonic() > deadline:
            return False
        i, cnt = mrv_index()
        if i == -1:
            return True
        if cnt == 0:
            return False
        last = rows[i][-1]
        # collect feasible candidates, shuffle for diversity
        cands = []
        for v in range(last + 1, scope + 1):
            if not any((v - a) in used for a in rows[i]):
                cands.append(v)
        rng.shuffle(cands)
        for v in cands:
            newdiffs = {v - a for a in rows[i]}
            if newdiffs & used:
                continue
            rows[i].append(v)
            used |= newdiffs
            remaining[i] -= 1
            if solve(depth + 1):
                return True
            rows[i].pop()
            used -= newdiffs
            remaining[i] += 1
        return False

    if solve(0):
        return rows
    return None
