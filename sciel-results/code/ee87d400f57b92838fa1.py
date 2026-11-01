import time
import random
import math

_DEADLINE = 112.0  # safety margin vs 120 s kill
_t0 = time.time()
_rng = random.Random(12345)


def _validity(rows):
    seen = set()
    for row in rows:
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ai
                if d in seen:
                    return False
                seen.add(d)
    return True


def _greedy_build(n_rows, k, seed):
    r = random.Random(seed)
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


def _local_search(n_rows, k, rows, deadline):
    rows = [list(r) for r in rows]
    rows.sort(key=lambda r: r[-1])
    used = set()
    for row in rows:
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                used.add(row[j] - ai)

    def scope():
        return max(r[-1] for r in rows)

    best = scope()
    best_rows = [list(r) for r in rows]
    n = k + 1
    cur = scope()
    while time.time() < deadline:
        improved = False
        # pick a row to work on: worst (largest) first, then random
        if _rng.random() < 0.6:
            idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            idx = _rng.randrange(n_rows)
        row = rows[idx]
        # try to shrink the largest mark
        target = row[-1]
        for cand in range(target - 1, 0, -1):
            if cand in row:
                continue
            newrow = row[:-1] + [cand]
            newrow.sort()
            if newrow[0] != 0:
                continue
            # diffs of this row
            rdiffs = set()
            for i in range(n):
                ai = newrow[i]
                for j in range(i + 1, n):
                    rdiffs.add(newrow[j] - ai)
            # check collision with other rows' diffs
            if any(d in used for d in rdiffs):
                continue
            # accept
            used.difference_update({row[j] - row[i] for i in range(n) for j in range(i + 1, n)})
            rows[idx] = newrow
            used.update(rdiffs)
            newscope = scope()
            if newscope < best:
                best = newscope
                best_rows = [list(r) for r in rows]
            cur = newscope
            improved = True
            break
        if not improved:
            # try a small random perturbation to escape local minima
            if _rng.random() < 0.5:
                # random re-greedy of one row
                other_used = set()
                for ri in range(n_rows):
                    if ri == idx:
                        continue
                    rr = rows[ri]
                    for i in range(n):
                        ai = rr[i]
                        for j in range(i + 1, n):
                            other_used.add(rr[j] - ai)
                base = rows[idx][:-1]  # keep all but last
                # rebuild last mark greedily
                row2 = base[:]
                x = base[-1]
                found = False
                while not found:
                    x += 1
                    new = {x - a for a in row2}
                    if not (new & other_used):
                        row2.append(x)
                        found = True
                if row2[-1] < rows[idx][-1]:
                    used.difference_update({rows[idx][j] - rows[idx][i] for i in range(len(rows[idx])) for j in range(i + 1, len(rows[idx]))})
                    rows[idx] = row2
                    for i in range(n):
                        ai = row2[i]
                        for j in range(i + 1, n):
                            used.add(row2[j] - ai)
                else:
                    # revert
                    pass
    return best_rows


def construct(n_rows, k):
    global _t0
    deadline = _t0 + _DEADLINE

    # baseline greedy
    rows = _greedy_build(n_rows, k, 0)
    best = max(r[-1] for r in rows)
    best_rows = [list(r) for r in rows]

    # multi-start randomized greedy + local search
    seed = 1
    while time.time() < deadline:
        cand = _greedy_build(n_rows, k, seed)
        s = max(r[-1] for r in cand)
        if s < best:
            best = s
            best_rows = [list(r) for r in cand]
        # local search on the better of current best and candidate
        start = cand if s < max(r[-1] for r in best_rows) else best_rows
        sub_deadline = min(deadline, time.time() + 6.0)
        ls = _local_search(n_rows, k, start, sub_deadline)
        s2 = max(r[-1] for r in ls)
        if s2 < best:
            best = s2
            best_rows = [list(r) for r in ls]
        seed += 1

    # final polish: try to shave 1 off each row's max
    for _ in range(3):
        changed = False
        for i in range(n_rows):
            row = best_rows[i]
            for cand in range(row[-1] - 1, 0, -1):
                if cand in row:
                    continue
                newrow = sorted(row[:-1] + [cand])
                if newrow[0] != 0:
                    continue
                # validate whole set
                trial = [list(r) for r in best_rows]
                trial[i] = newrow
                if _validity(trial):
                    best_rows[i] = newrow
                    changed = True
                    break
        if not changed:
            break

    return best_rows
