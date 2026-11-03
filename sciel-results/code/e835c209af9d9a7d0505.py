import time
import random

def construct(n_rows, k):
    deadline = time.time() + 145.0   # explicit safety margin under the 150 s limit
    rng = random.Random(12345)
    n_diff = k * (k + 1) // 2        # 15 for k=6
    k1 = k + 1

    def gaps_to_row(gaps):
        row = [0]
        s = 0
        for g in gaps:
            s += g
            row.append(s)
        return row

    def row_diffs(row):
        return [row[j] - row[i] for i in range(k1) for j in range(i + 1, k1)]

    def max_entry(rows):
        return max(r[-1] for r in rows)

    # ---- greedy baseline (fast, always valid) ----
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    best = [r[:] for r in rows]
    best_val = max_entry(best)

    # ---- local search over rows (permutation invariance + smaller gap sequences) ----
    while time.time() < deadline:
        # random row order to escape the greedy's fixed ordering
        order = list(range(n_rows))
        rng.shuffle(order)
        used, cand = set(), []
        ok = True
        for idx in order:
            orig = best[idx]
            row = [0]
            cur = 0
            placed = 0
            while placed < k:
                placed += 1
                cur += 1
                new = {cur - a for a in row}
                if not new & used:
                    row.append(cur)
                    used |= new
            cand.append(row)
        # try to shrink each row's gap sequence (lower max entry + frees big diffs)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n_rows):
                row = cand[i]
                diffs = row_diffs(row)
                for d in diffs:
                    used.discard(d)
                gaps = [row[j + 1] - row[j] for j in range(k)]
                row2 = [0]
                cur = 0
                for g in gaps:
                    cur += g
                    row2.append(cur)
                d2 = row_diffs(row2)
                if not (set(d2) & used):
                    cand[i] = row2
                    used.update(d2)
                    improved = True
        val = max_entry(cand)
        if val < best_val:
            best = [r[:] for r in cand]
            best_val = val
            if best_val <= 105:   # counting bound: cannot do better
                break

    return best
