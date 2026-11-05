"""Difference triangle sets (n_rows, k): minimize the largest entry.

Combines a multi-strategy greedy construction with a simulated-annealing
local search that uses rebuild / gap-fill moves and reshuffle (worse) moves
to escape local minima, followed by a two-phase scope-tightening pass.
"""
import time
import random
import math
import numpy as np

_TIME_LIMIT = 120.0
_DEADLINE = None
_DEADLINE_SET = False


def _time_left():
    global _DEADLINE, _DEADLINE_SET
    if not _DEADLINE_SET:
        _DEADLINE = time.time() + _TIME_LIMIT
        _DEADLINE_SET = True
    return max(0.0, _DEADLINE - time.time())


def _validate(rows):
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s


def _all_diffs_except(rows, skip_idx):
    s = set()
    for idx, row in enumerate(rows):
        if idx == skip_idx:
            continue
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
    return s


# ------------------------- greedy construction -------------------------
def _greedy_row(used, k, rng, strategy):
    row = [0]
    x = 0
    while len(row) < k + 1:
        if strategy == "pure":
            x += 1
            while True:
                new = {x - a for a in row}
                if not new & used:
                    break
                x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
        elif strategy == "shuffled":
            base = x + 1
            window = 50
            candidates = list(range(base, base + window))
            if rng:
                rng.shuffle(candidates)
            found = False
            for cand in candidates:
                new = {cand - a for a in row}
                if not new & used:
                    row.append(cand)
                    used |= {cand - a for a in row[:-1]}
                    x = cand
                    found = True
                    break
            if not found:
                cand = base + window
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= {cand - a for a in row[:-1]}
                        x = cand
                        break
                    cand += 1
        elif strategy == "lookahead":
            base = x + 1
            valid_cands = []
            for cand in range(base, base + 50):
                new = {cand - a for a in row}
                if not new & used:
                    valid_cands.append(cand)
                if len(valid_cands) >= 8:
                    break
            if not valid_cands:
                cand = base
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        valid_cands.append(cand)
                        break
                    cand += 1
            best_cand = valid_cands[0]
            best_score = -1
            for cand in valid_cands[:6]:
                new = {cand - a for a in row}
                next_count = 0
                for nc in range(cand + 1, cand + 25):
                    nd = {nc - a for a in row}
                    nd.add(nc - cand)
                    if not (nd & (used | new)):
                        next_count += 1
                    if next_count >= 4:
                        break
                score = next_count * 100 - cand
                if score > best_score:
                    best_score = score
                    best_cand = cand
            row.append(best_cand)
            used |= {best_cand - a for a in row[:-1]}
            x = best_cand
        elif strategy == "pressure":
            base = x + 1
            valid_cands = []
            for cand in range(base, base + 60):
                new = {cand - a for a in row}
                if not new & used:
                    valid_cands.append((cand, max(new)))
                if len(valid_cands) >= 10:
                    break
            if not valid_cands:
                cand = base
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        valid_cands.append((cand, max(new)))
                        break
                    cand += 1
            valid_cands.sort(key=lambda t: (t[1], t[0]))
            best_cand = valid_cands[0][0]
            row.append(best_cand)
            used |= {best_cand - a for a in row[:-1]}
            x = best_cand
    return row


def _construct_diverse(n_rows, k, rng, strategy):
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(_greedy_row(used, k, rng, strategy))
    return rows


# --------------------------- vectorized helpers ---------------------------
def _row_vec_diffs(row):
    a = np.asarray(row, dtype=np.int64)
    return np.unique(a[np.newaxis, :] - a[:, np.newaxis])


def _row_conflicts(row, other):
    a = np.asarray(row, dtype=np.int64)
    o = np.asarray(other, dtype=np.int64)
    m = np.unique(a[np.newaxis, :] - a[:, np.newaxis])
    return np.intersect1d(m, o, assume_unique=False)


def _row_internal_ok(row):
    a = np.asarray(row, dtype=np.int64)
    d = np.unique(a[np.newaxis, :] - a[:, np.newaxis])
    return len(d) == len(row) * (len(row) - 1) // 2


def _row_internal_conflicts(row, other):
    a = np.asarray(row, dtype=np.int64)
    o = np.asarray(other, dtype=np.int64)
    m = np.unique(a[np.newaxis, :] - a[:, np.newaxis])
    return np.intersect1d(m, o, assume_unique=False)


# --------------------------- backtracking rebuild ---------------------------
def _backtrack_row(used, k, max_scope, rng):
    row = [0]

    def bt():
        if len(row) == k + 1:
            return True
        start = row[-1] + 1
        hi = min(max_scope, start + 60)
        candidates = list(range(start, hi + 1))
        if rng and len(candidates) > 10:
            rng.shuffle(candidates[5:])
        for x in candidates:
            if x > max_scope:
                break
            new = {x - a for a in row}
            if new & used:
                continue
            row.append(x)
            used.update(new)
            if bt():
                return True
            row.pop()
            used.difference_update(new)
        return False

    return row if bt() else None


# --------------------------- simulated annealing ---------------------------
def _sa_local_search(rows, n_rows, k, rng, max_iters=200000):
    rows = [list(r) for r in rows]
    n = k + 1
    row_diffs = [_row_diffs(r) for r in rows]
    all_diffs = set().union(*row_diffs) if row_diffs else set()
    all_diffs.discard(0)
    other = [all_diffs - row_diffs[i] for i in range(n_rows)]

    best_scope = _scope(rows)
    best_rows = [r[:] for r in rows]
    best_cost = sum(len(r) for r in rows)  # sum of row maxima (proxy)
    cur_cost = best_cost

    start_t = time.time()
    t0 = 200.0
    t1 = 3.0
    total = max(1, max_iters)

    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        frac = it / total
        temp = t0 * (t1 / t0) ** frac

        r = rng.randrange(n_rows)
        row = rows[r]
        od = other[r]

        # --- move selection ---
        if rng.random() < 0.35:
            # full rebuild (with small bias toward smaller values)
            new_row = None
            for _try in range(3):
                cand = _backtrack_row(od, k, row[-1] + 10, rng)
                if cand is not None:
                    new_row = cand
                    break
            if new_row is None:
                continue
        elif rng.random() < 0.5:
            # gap fill: add a value into a random gap
            pos = rng.randrange(1, len(row))
            lo = row[pos - 1] + 1
            hi = row[pos] - 1
            if hi < lo:
                continue
            new_row = row[:]
            new_row.insert(pos, rng.randrange(lo, hi + 1))
            if not _row_internal_ok(new_row):
                continue
            if _row_internal_conflicts(new_row, od).size:
                continue
        else:
            # gap shrink: remove a random interior value
            pos = rng.randrange(1, len(row) - 1)
            new_row = row[:pos] + row[pos + 1:]

        if len(new_row) != n:
            continue
        if not _row_internal_ok(new_row):
            continue
        if _row_internal_conflicts(new_row, od).size:
            continue

        new_cost = sum(len(x) for x in rows) - len(row) + len(new_row)
        delta = new_cost - cur_cost

        # accept?
        if delta <= 0 or (temp > 1e-6 and rng.random() < math.exp(-delta / temp)):
            # commit
            all_diffs.difference_update(row_diffs[r])
            all_diffs.update(_row_diffs(new_row))
            rows[r] = new_row
            row_diffs[r] = _row_diffs(new_row)
            for j in range(n_rows):
                if j != r:
                    other[j] = all_diffs - row_diffs[j]
            other[r] = all_diffs - row_diffs[r]
            cur_cost = new_cost
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [x[:] for x in rows]
                best_cost = cur_cost
                if best_scope <= 112:
                    break

    return best_rows, best_scope


# --------------------------- scope tightening ---------------------------
def _tighten_scope(rows, n_rows, k, rng):
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)

    for _outer in range(60):
        if _time_left() < 12.0:
            break
        target = best_scope - 1
        if target < k + 1:
            break
        # try to rebuild the max row under target
        max_idx = max(range(n_rows), key=lambda i: best_rows[i][-1])
        used_other = _all_diffs_except(best_rows, max_idx)
        new_row = _backtrack_row(used_other, k, target, rng)
        if new_row is None:
            new_row = _backtrack_row(used_other, k, target, rng)
        if new_row is None:
            # greedy rebuild under target
            gr = _greedy_row(used_other, k, rng, "pure")
            if gr[-1] <= target:
                new_row = gr
        if new_row is not None and new_row[-1] <= target:
            best_rows[max_idx] = new_row
            ns = _scope(best_rows)
            if ns < best_scope:
                best_scope = ns
            else:
                break
        else:
            break
    return best_rows, best_scope


# ------------------------------- entry point -------------------------------
def construct(n_rows, k):
    global _DEADLINE, _DEADLINE_SET
    _DEADLINE = time.time() + _TIME_LIMIT
    _DEADLINE_SET = True

    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1) for _ in range(max(n_rows, 0))]

    rng = random.Random(42)

    strategies = ["pure", "shuffled", "lookahead", "pressure",
                  "shuffled", "lookahead", "pure", "pressure"]
    best_rows = None
    best_scope = float('inf')

    for strategy in strategies:
        if _time_left() < 15.0:
            break
        rows = _construct_diverse(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    attempts = 0
    while _time_left() > 25.0 and attempts < 60:
        attempts += 1
        strategy = rng.choice(["pure", "shuffled", "lookahead", "pressure"])
        rows = _construct_diverse(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            if best_scope <= 112:
                break

    # Simulated annealing local search (escape mechanism)
    if _time_left() > 20.0:
        rows_sa, scope_sa = _sa_local_search(best_rows, n_rows, k, rng, max_iters=150000)
        if scope_sa < best_scope:
            best_scope = scope_sa
            best_rows = rows_sa

    # Tighten scope
    if _time_left() > 15.0:
        rows_t, scope_t = _tighten_scope(best_rows, n_rows, k, rng)
        if scope_t < best_scope:
            best_scope = scope_t
            best_rows = rows_t

    # Extra restarts if time permits
    if _time_left() > 25.0:
        extra = 0
        while _time_left() > 15.0 and extra < 25:
            extra += 1
            strategy = rng.choice(["pressure", "shuffled", "lookahead"])
            rows = _construct_diverse(n_rows, k, rng, strategy)
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= 112:
                    break
            if _time_left() > 15.0:
                rows_sa, scope_sa = _sa_local_search(rows, n_rows, k, rng, max_iters=40000)
                if scope_sa < best_scope:
                    best_scope = scope_sa
                    best_rows = rows_sa
                    if best_scope <= 112:
                        break

    # Final local search pass
    if _time_left() > 10.0:
        rows_ls, scope_ls = _sa_local_search(best_rows, n_rows, k, rng, max_iters=50000)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls
        if _time_left() > 8.0:
            rows_t, scope_t = _tighten_scope(best_rows, n_rows, k, rng)
            if scope_t < best_scope:
                best_scope = scope_t
                best_rows = rows_t

    if not _validate(best_rows):
        used = set()
        rows = []
        for _ in range(n_rows):
            rows.append(_greedy_row(used, k, None, "pure"))
        return rows

    return best_rows
