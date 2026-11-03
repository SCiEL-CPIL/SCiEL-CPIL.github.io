"""Difference triangle set (7,5) -- simulated annealing with incremental diff counting."""
import time
import random

_DEADLINE = 145.0  # safety margin under the 150 s wall-clock kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _validate(rows):
    """Return True iff every positive intra-row difference is globally distinct."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ai
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _row_diffs(row):
    """Return the multiset (list) of positive differences inside one row."""
    out = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            out.append(row[j] - ai)
    return out


def _init_csp(n_rows, k):
    """Greedy forward-checking build of a valid starting solution."""
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


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0]] * n_rows

    rng = random.Random(12345)

    # --- initial feasible solution via forward checking ---
    rows = _init_csp(n_rows, k)

    # global difference multiset
    diff_count = {}
    for row in rows:
        for d in _row_diffs(row):
            diff_count[d] = diff_count.get(d, 0) + 1

    def conflicts():
        c = 0
        for v in diff_count.values():
            if v > 1:
                c += v - 1
        return c

    cur_conf = conflicts()
    cur_scope = _scope(rows)

    # objective = scope * BIG + conflicts  (validity dominates)
    BIG = 10 ** 9
    cur_obj = cur_scope * BIG + cur_conf

    best_rows = [r[:] for r in rows]
    best_obj = cur_obj
    best_scope = cur_scope

    # --- simulated annealing ---
    T = 2000.0
    T_min = 1.0
    cooling = 0.9995

    def propose():
        """Pick a row and a non-first slot; return (row_idx, pos, old_val)."""
        ri = rng.randrange(n_rows)
        row = rows[ri]
        pos = rng.randrange(1, len(row))  # never touch the leading 0
        return ri, pos, row[pos]

    def apply_move(ri, pos, new_val):
        """Change rows[ri][pos] to new_val; update diff_count. Return old_val."""
        row = rows[ri]
        old = row[pos]
        # remove old differences involving this position
        for i in range(len(row)):
            if i == pos:
                continue
            d = row[pos] - row[i] if pos > i else row[i] - row[pos]
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        row[pos] = new_val
        # add new differences
        for i in range(len(row)):
            if i == pos:
                continue
            d = row[pos] - row[i] if pos > i else row[i] - row[pos]
            diff_count[d] = diff_count.get(d, 0) + 1
        return old

    def revert_move(ri, pos, old_val):
        row = rows[ri]
        # remove current differences
        for i in range(len(row)):
            if i == pos:
                continue
            d = row[pos] - row[i] if pos > i else row[i] - row[pos]
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        row[pos] = old_val
        for i in range(len(row)):
            if i == pos:
                continue
            d = row[pos] - row[i] if pos > i else row[i] - row[pos]
            diff_count[d] = diff_count.get(d, 0) + 1

    step = 0
    while T > T_min and _elapsed() < _DEADLINE:
        ri, pos, old_val = propose()
        row = rows[ri]

        # choose a candidate new value that keeps the row strictly increasing
        lo = row[pos - 1] + 1
        hi = row[pos + 1] - 1 if pos + 1 < len(row) else max(old_val + 6, old_val + 1)
        if lo > hi:
            continue
        # bias toward compact (small) values, but allow exploration
        span = hi - lo + 1
        if span <= 40 or rng.random() < 0.5:
            new_val = rng.randrange(lo, hi + 1)
        else:
            # skewed toward the low end for compactness
            new_val = lo + int((span - 1) * (rng.random() ** 2))

        if new_val == old_val:
            continue

        apply_move(ri, pos, new_val)
        new_conf = conflicts()
        new_scope = _scope(rows)
        new_obj = new_scope * BIG + new_conf
        delta = new_obj - cur_obj

        if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
            cur_obj = new_obj
            cur_conf = new_conf
            cur_scope = new_scope
            if new_obj < best_obj:
                best_obj = new_obj
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
        else:
            revert_move(ri, pos, old_val)

        T *= cooling
        step += 1

    # --- final: if best is valid, return it; otherwise return best valid seen ---
    if _validate(best_rows):
        return best_rows

    # Fallback: the initial CSP solution is always valid
    return _init_csp(n_rows, k)
