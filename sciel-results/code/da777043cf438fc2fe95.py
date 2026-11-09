import time
import random

_DEADLINE = 150.0
_MARGIN = 20.0  # seconds of safety before the hard 150s kill


def _all_diffs(row):
    """Return the set of positive differences within a single row."""
    out = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            out.add(row[j] - ai)
    return out


def _scope(rows):
    return max(r[-1] for r in rows)


def _collision_count(rows, used_map):
    """Number of difference values used by more than one row (sum over rows of extra uses)."""
    # used_map: dict value -> count of rows containing it
    # collisions = sum over values of (count-1) for count>=2, but we count per-row extras.
    total = 0
    for c in used_map.values():
        if c > 1:
            total += (c - 1)
    return total


def _rebuild_used_map(rows):
    used_map = {}
    for r in rows:
        for d in _all_diffs(r):
            used_map[d] = used_map.get(d, 0) + 1
    return used_map


def _random_row(k, lo, hi):
    """Random strictly increasing row of k+1 ints starting at 0, all <= hi."""
    # choose k distinct values in [1, hi]
    if hi < k:
        # fallback: minimal row
        return list(range(k + 1))
    vals = random.sample(range(1, hi + 1), k)
    vals.sort()
    return [0] + vals


def _nudge_row(row, k, hi, amount):
    """Perturb a row: keep 0, randomly adjust some marks, keep increasing, cap at hi."""
    new = [0] + list(row[1:])
    for idx in range(1, len(new)):
        if random.random() < 0.5:
            delta = random.randint(-amount, amount)
            new[idx] += delta
    # enforce strict increase and bounds
    fixed = [0]
    for v in new[1:]:
        v = max(fixed[-1] + 1, 1)
        v = min(v, hi)
        fixed.append(v)
    # ensure strictly increasing (dedupe by pushing up)
    out = [0]
    for v in fixed[1:]:
        if v <= out[-1]:
            v = out[-1] + 1
        out.append(v)
    return out


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN
    n = k + 1  # entries per row

    # ---- initial population: greedy-ish random rows, then local repair ----
    def greedy_row(used):
        row = [0]
        x = 0
        while len(row) < n:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        return row

    # build initial rows greedily to get a decent starting scope
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(greedy_row(used))
    used_map = _rebuild_used_map(rows)
    best_scope = _scope(rows)
    best_rows = [r[:] for r in rows]

    # If already collision-free and small, still try to improve scope via SA.
    lam = 1.0  # weight on collisions

    def cost(rows_, umap):
        return _scope(rows_) + lam * _collision_count(rows_, umap)

    cur_cost = cost(rows, used_map)
    T = 30.0
    T_min = 0.05
    cooling = 0.995

    rng = random.Random(12345)

    while time.time() < deadline:
        # pick a move type
        r = rng.random()
        if r < 0.4:
            # rebuild a random row from scratch
            i = rng.randrange(n_rows)
            hi = best_scope + 5
            old = rows[i]
            newrow = _random_row(k, 1, hi)
            # update used_map
            for d in _all_diffs(old):
                used_map[d] -= 1
                if used_map[d] == 0:
                    del used_map[d]
            for d in _all_diffs(newrow):
                used_map[d] = used_map.get(d, 0) + 1
            rows[i] = newrow
            new_cost = cost(rows, used_map)
            # accept
            if new_cost <= cur_cost or rng.random() < pow(2.718281828, -(new_cost - cur_cost) / T):
                cur_cost = new_cost
                if _collision_count(rows, used_map) == 0:
                    s = _scope(rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = [x[:] for x in rows]
            else:
                # revert
                rows[i] = old
                for d in _all_diffs(old):
                    used_map[d] = used_map.get(d, 0) + 1
                for d in _all_diffs(newrow):
                    used_map[d] -= 1
                    if used_map[d] == 0:
                        del used_map[d]
        else:
            # nudge a random row
            i = rng.randrange(n_rows)
            old = rows[i]
            hi = best_scope + 5
            newrow = _nudge_row(old, k, hi, amount=3)
            for d in _all_diffs(old):
                used_map[d] -= 1
                if used_map[d] == 0:
                    del used_map[d]
            for d in _all_diffs(newrow):
                used_map[d] = used_map.get(d, 0) + 1
            rows[i] = newrow
            new_cost = cost(rows, used_map)
            if new_cost <= cur_cost or rng.random() < pow(2.718281828, -(new_cost - cur_cost) / T):
                cur_cost = new_cost
                if _collision_count(rows, used_map) == 0:
                    s = _scope(rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = [x[:] for x in rows]
            else:
                rows[i] = old
                for d in _all_diffs(old):
                    used_map[d] = used_map.get(d, 0) + 1
                for d in _all_diffs(newrow):
                    used_map[d] -= 1
                    if used_map[d] == 0:
                        del used_map[d]

        T = max(T_min, T * cooling)

    # return best valid (collision-free) rows found
    return best_rows
