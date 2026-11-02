"""Construct difference triangle set via randomized greedy + local improvement."""
import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin under 150s

    # Number of differences per row: k+1 choose 2
    diffs_per_row = (k + 1) * k // 2

    def row_diffs(row):
        """All positive differences in a row."""
        out = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                out.add(row[j] - row[i])
        return out

    def is_valid(rows):
        used = set()
        for row in rows:
            d = row_diffs(row)
            if d & used:
                return False
            used |= d
        return True

    def get_max_val(rows):
        return max(max(r) for r in rows)

    def greedy_build(used, k, order=None):
        """Build one row greedily. order: optional priority for candidate x (list of x to try first)."""
        row = [0]
        while len(row) < k + 1:
            found = False
            # Try x from 1 upward, but we can bias by trying a preferred set first
            if order is not None:
                # try preferred candidates first (sorted by index in order), then fallback
                pref = {v: i for i, v in enumerate(order)}
                # generate candidates in increasing x, but sort by (pref.get(x, 10**9), x)
                # We need to search x from 1..; we'll just iterate x and pick those in pref first
                # Simpler: build a candidate list of x in 1..some bound, sort by key
                # But we don't know bound; iterate x increasing, but if x in pref and pref index small, prefer
                # Actually to keep it simple and correct, just iterate x increasing; 'order' used to
                # pre-allocate a bias by trying a shuffled small range first? Let's keep greedy pure
                # and handle ordering at a higher level.
                pass
            x = row[-1]
            while True:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
                    found = True
                    break
            if not found:
                break
        return row

    def randomized_greedy(n_rows, k, seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # bias: occasionally skip a candidate to explore different shapes
            while len(row) < k + 1:
                x = row[-1]
                while True:
                    x += 1
                    # small probability to skip this x to diversify
                    if rng.random() < 0.05:
                        continue
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                        break
            rows.append(row)
        return rows

    def local_improve(rows, k, max_iters=200):
        """Try to reduce the max value by adjusting rows one at a time."""
        rows = [list(r) for r in rows]
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for idx in range(len(rows)):
                # try to rebuild this row with a tighter upper bound
                others_used = set()
                for j, r in enumerate(rows):
                    if j != idx:
                        others_used |= row_diffs(r)
                old_max = max(rows[idx])
                # try to find a new row with smaller max
                best_row = rows[idx]
                best_max = old_max
                # try greedy with decreasing upper bound
                for upper in range(old_max - 1, 0, -1):
                    # try to build a row within [0, upper]
                    row = [0]
                    used = set(others_used)
                    ok = True
                    while len(row) < k + 1:
                        found = False
                        x = row[-1]
                        while x < upper:
                            x += 1
                            new = {x - a for a in row}
                            if not (new & used):
                                row.append(x)
                                used |= new
                                found = True
                                break
                        if not found:
                            ok = False
                            break
                    if ok and max(row) < best_max:
                        best_row = row
                        best_max = max(row)
                        # we can stop early since we're going downward
                        break
                if best_max < old_max:
                    rows[idx] = best_row
                    improved = True
        return rows

    # Initial solution
    best_rows = None
    best_val = float('inf')

    # Try a few seeds of randomized greedy
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        rows = randomized_greedy(n_rows, k, seed)
        if is_valid(rows):
            v = get_max_val(rows)
            if v < best_val:
                best_val = v
                best_rows = [list(r) for r in rows]

    # Local improvement on the best
    if best_rows is not None:
        if time.time() < deadline - 5:
            improved = local_improve(best_rows, k)
            if is_valid(improved):
                v = get_max_val(improved)
                if v < best_val:
                    best_val = v
                    best_rows = improved

    # If no valid solution found (shouldn't happen), fall back to pure greedy
    if best_rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1]
                while True:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                        break
            rows.append(row)
        best_rows = rows

    return best_rows
