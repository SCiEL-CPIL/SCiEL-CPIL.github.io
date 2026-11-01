import random
import time

# EVOLVE-BLOCK-START

def _valid(rows, n_rows, k):
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(len(r)):
            if i > 0 and r[i] <= r[i - 1]:
                return False
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in used:
                    return False
                used.add(d)
    return len(rows) == n_rows

def _scope(rows):
    return max(max(r) for r in rows)

def _greedy(rows, n_rows, k, rng, max_probe):
    used = set()
    for idx, r in enumerate(rows):
        if len(r) != k + 1:
            continue
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                used.add(r[j] - r[i])
    for idx in range(n_rows):
        if idx < len(rows) and len(rows[idx]) == k + 1:
            continue
        row = [0]
        for step in range(1, k + 1):
            start = 1
            placed = False
            for _ in range(max_probe):
                x = start + rng.randrange(max_probe)
                if x in row:
                    continue
                ok = True
                for a in row:
                    d = abs(x - a)
                    if d in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(abs(x - a))
                    placed = True
                    break
            if not placed:
                # deterministic fallback scan
                x = 1
                while True:
                    if x in row:
                        x += 1
                        continue
                    ok = True
                    for a in row:
                        if abs(x - a) in used:
                            ok = False
                            break
                    if ok:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(abs(x - a))
                        placed = True
                        break
                    x += 1
        if len(row) == k + 1:
            rows[idx] = row
    return rows

def _trim(rows, n_rows, k, rng, deadline):
    used = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                used.add(r[j] - r[i])
    scope = _scope(rows)
    # Try to lower the largest entry by replacing the last element of a row with a smaller one.
    for _ in range(200000):
        if time.time() >= deadline:
            break
        idx = rng.randrange(n_rows)
        r = rows[idx]
        last = r[-1]
        # candidate smaller values
        cand = []
        for _ in range(12):
            c = rng.randrange(1, last)
            cand.append(c)
        cand.append(last - 1)
        rng.shuffle(cand)
        for c in cand:
            if c in r[:-1] or c <= r[-2]:
                continue
            # remove last's diffs
            for a in r[:-1]:
                used.discard(last - a)
            ok = True
            for a in r[:-1]:
                d = c - a
                if d in used:
                    ok = False
                    break
            if ok:
                for a in r[:-1]:
                    used.add(c - a)
                r[-1] = c
                scope = min(scope, c)
                break
            else:
                # restore
                for a in r[:-1]:
                    used.add(last - a)
                break
    return scope

def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + 110.0
    rng = random.Random(1234567)

    best = None
    best_scope = None

    # Initial random greedy constructions (several tries)
    for attempt in range(30):
        if time.time() >= deadline:
            break
        rows = [[0] for _ in range(n_rows)]
        _greedy(rows, n_rows, k, rng, max_probe=80)
        if not _valid(rows, n_rows, k):
            continue
        s = _scope(rows)
        if best_scope is None or s < best_scope:
            best_scope = s
            best = [list(r) for r in rows]
        # local search trimming
        s2 = _trim(rows, n_rows, k, rng, deadline)
        if _valid(rows, n_rows, k):
            s2 = _scope(rows)
            if best_scope is None or s2 < best_scope:
                best_scope = s2
                best = [list(r) for r in rows]

    if best is None:
        # deterministic fallback: plain greedy
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
        best = rows
        best_scope = _scope(rows)

    return best

# EVOLVE-BLOCK-END
