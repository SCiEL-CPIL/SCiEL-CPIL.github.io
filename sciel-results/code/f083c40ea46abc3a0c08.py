"""Difference triangle set (7,5): minimize the largest entry via randomized local search."""
import random
import time

# EVOLVE-BLOCK-START
def _check_valid(rows, k, n_rows):
    used = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(k + 1):
            a = row[i]
            for j in range(i + 1, k + 1):
                d = row[j] - a
                if d in used:
                    return False
                used.add(d)
    return True


def _build(rows, k):
    used = set()
    last = 0
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return None
        for i in range(1, k + 1):
            a = row[i]
            if a <= last:
                return None
            last = a
            for j in range(i):
                d = a - row[j]
                if d in used:
                    return None
                used.add(d)
    return last


def construct(n_rows, k):
    deadline = time.time() + 110.0
    rng = random.Random(123456789)
    total = n_rows * k  # number of non-zero marks

    def rand_row():
        r = [0]
        for _ in range(k):
            r.append(r[-1] + rng.randint(1, 40))
        return r

    best = None
    best_val = float('inf')

    def consider(rows):
        nonlocal best, best_val
        v = _build(rows, k)
        if v is not None and v < best_val:
            best_val = v
            best = [list(x) for x in rows]

    # Seed with greedy (known-good baseline)
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
    consider(rows)

    # Random local search
    rows = [rand_row() for _ in range(n_rows)]
    consider(rows)
    cur = rows

    temp = 40.0
    t0 = time.time()
    steps = 0
    while time.time() < deadline - 2.0:
        steps += 1
        ri = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        old = cur[ri][pos]
        new = old + rng.randint(-15, 15)
        if new <= cur[ri][pos - 1]:
            continue
        cur[ri][pos] = new

        v = _build(cur, k)
        if v is None:
            # reject (restore)
            cur[ri][pos] = old
        else:
            consider(cur)
            if v <= best_val:
                # accept improvement
                pass
            else:
                # maybe accept worse
                d = v - best_val
                if rng.random() < (temp / max(1, d)):
                    pass
                else:
                    cur[ri][pos] = old

        if steps % 400 == 0:
            temp *= 0.999
            if temp < 0.5:
                temp = 0.5

    if best is None:
        # fallback: regenerate greedy
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
    return best
# EVOLVE-BLOCK-END
