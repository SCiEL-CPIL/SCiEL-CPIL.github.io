"""Difference triangle set (4,7): minimize the largest entry (scope).

Hybrid: fast greedy baseline + time-budgeted local search (row rebuilds and
mark-swap moves) with an explicit deadline guard. All collision tests use a
global difference bitset (Python int) for O(1) checks.
"""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 150.0
_MARGIN = 3.0  # seconds of safety margin before the hard kill


def _diff_mask(row):
    """Bitmask of all positive differences within a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            m |= 1 << (row[j] - ai)
    return m


def _row_collides(row, used):
    """True if any positive difference of row is already in used."""
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            if (used >> (row[j] - ai)) & 1:
                return True
    return False


def _greedy(n_rows, k, rng, bias):
    """Greedy ruler construction. bias>0 adds randomization to escape local optima."""
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if bias and rng.random() < bias:
                x += rng.randint(0, 2)
            ok = True
            for a in row:
                d = x - a
                if (used >> d) & 1:
                    ok = False
                    break
            if ok:
                row.append(x)
                for a in row[:-1]:
                    used |= 1 << (x - a)
        rows.append(row)
    return rows


def _scope(rows):
    return max(max(r) for r in rows)


def _build_used(rows):
    used = 0
    for r in rows:
        used |= _diff_mask(r)
    return used


def _rebuild_row(row, used, rng, max_scope):
    """Try to rebuild a single row (keeping it inside max_scope) to avoid used diffs.
    Returns a new valid row or None."""
    k1 = len(row)
    target = row[-1]
    # Randomized greedy fill with backtracking-lite: try several random orders.
    for _attempt in range(40):
        cand = [0]
        x = 0
        ok = True
        while len(cand) < k1:
            x += 1
            if x > max_scope:
                ok = False
                break
            # occasional random jump to diversify
            if rng.random() < 0.15:
                x += rng.randint(0, 3)
            if x > max_scope:
                ok = False
                break
            good = True
            for a in cand:
                if (used >> (x - a)) & 1:
                    good = False
                    break
            if good:
                cand.append(x)
        if ok and len(cand) == k1:
            # verify no internal collision (differences within cand are new to this row,
            # but must not collide with each other? they are all within one row and are
            # required distinct globally; check against used only is enough for cross-row,
            # but within-row duplicates must also be avoided -> check full mask not in used)
            if not (_diff_mask(cand) & used):
                return cand
    return None


def _try_swap(rows, masks, used, idx, j, x, rng):
    """Move mark j of row idx to value x (keeping sorted, distinct, within scope).
    Returns (new_used, new_mask, new_row) or None if invalid."""
    row = rows[idx]
    k1 = len(row)
    if x == row[j]:
        return None
    # new row with mark j replaced by x, must stay strictly increasing and unique
    new_row = row[:]
    new_row[j] = x
    if new_row[j - 1] >= x or (j + 1 < k1 and new_row[j + 1] <= x):
        return None
    # also ensure no duplicate with neighbours handled above; check full strict increase
    for t in range(1, k1):
        if new_row[t] <= new_row[t - 1]:
            return None
    new_mask = _diff_mask(new_row)
    # remove old mask, add new; check new_mask doesn't collide with (used - old_mask)
    remaining = used & ~masks[idx]
    if new_mask & remaining:
        return None
    if new_mask & masks[idx]:
        # internal duplicates within the new row itself
        return None
    new_used = remaining | new_mask
    return new_used, new_mask, new_row


def construct(n_rows, k):
    start = time.time()
    rng = random.Random(12345)

    # 1) Fast baseline (deterministic greedy)
    base = _greedy(n_rows, k, rng, bias=0.0)
    best = [r[:] for r in base]
    best_scope = _scope(best)

    deadline = start + _DEADLINE - _MARGIN

    # 2) Local search: row rebuilds + mark swaps, keep best valid scope.
    def _run_search():
        nonlocal best, best_scope
        cur = [r[:] for r in base]
        cur_scope = best_scope
        masks = [_diff_mask(r) for r in cur]
        used = _build_used(cur)

        while time.time() < deadline:
            # --- Row rebuild phase: try to shrink a row's max (and thus scope) ---
            improved = False
            for idx in range(n_rows):
                if time.time() >= deadline:
                    break
                row = cur[idx]
                target = row[-1]
                # try to rebuild with a smaller max
                for new_max in range(target - 1, max(0, target - 60), -1):
                    nr = _rebuild_row(row, used & ~masks[idx], rng, new_max)
                    if nr is not None and _diff_mask(nr) & (used & ~masks[idx]) == 0:
                        new_mask = _diff_mask(nr)
                        used = (used & ~masks[idx]) | new_mask
                        masks[idx] = new_mask
                        cur[idx] = nr
                        ns = _scope(cur)
                        if ns < cur_scope:
                            cur_scope = ns
                        improved = True
                        break
            if cur_scope < best_scope:
                best = [r[:] for r in cur]
                best_scope = cur_scope

            # --- Mark swap phase: try to reduce scope by lowering the max marks ---
            # Identify rows holding the current max and try to pull their last mark down.
            for _ in range(300):
                if time.time() >= deadline:
                    break
                # pick a row with a high last mark
                order = list(range(n_rows))
                rng.shuffle(order)
                done = False
                for idx in order:
                    if done:
                        break
                    row = cur[idx]
                    last = row[-1]
                    # try to lower the last mark
                    for x in range(last - 1, max(0, last - 25), -1):
                        res = _try_swap(cur, masks, used, idx, len(row) - 1, x, rng)
                        if res is not None:
                            new_used, new_mask, new_row = res
                            used = new_used
                            masks[idx] = new_mask
                            cur[idx] = new_row
                            ns = _scope(cur)
                            if ns < cur_scope:
                                cur_scope = ns
                            done = True
                            break
                    if cur_scope < best_scope:
                        best = [r[:] for r in cur]
                        best_scope = cur_scope
                        done = True

            # --- Diversify: random perturbation of one row to escape local minima ---
            if rng.random() < 0.5:
                idx = rng.randrange(n_rows)
                nr = _rebuild_row(cur[idx], used & ~masks[idx], rng, cur_scope)
                if nr is not None and _diff_mask(nr) & (used & ~masks[idx]) == 0:
                    new_mask = _diff_mask(nr)
                    used = (used & ~masks[idx]) | new_mask
                    masks[idx] = new_mask
                    cur[idx] = nr

            # Occasionally restart from a fresh greedy to explore a different basin
            if rng.random() < 0.05:
                fresh = _greedy(n_rows, k, rng, bias=0.02)
                fs = _scope(fresh)
                if fs < cur_scope:
                    cur = [r[:] for r in fresh]
                    cur_scope = fs
                    masks = [_diff_mask(r) for r in cur]
                    used = _build_used(cur)
                    if cur_scope < best_scope:
                        best = [r[:] for r in cur]
                        best_scope = cur_scope

    try:
        _run_search()
    except Exception:
        pass

    return best
# EVOLVE-BLOCK-END
