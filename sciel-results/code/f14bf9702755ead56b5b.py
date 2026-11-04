"""Fresh design: randomized restarts with backtracking + local search, with a hard deadline guard."""
import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 145.0  # safety margin under 150s kill
    best_rows = None
    best_scope = float('inf')

    def time_left():
        return deadline - time.time()

    # Greedy baseline (deterministic) to always have a valid fallback
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in rows)

    # If greedy already achieves the counting bound, return immediately
    # counting bound: total diffs = n_rows * k*(k+1)//2, must fit in 1..scope
    total_diffs = n_rows * k * (k + 1) // 2
    if best_scope * (best_scope + 1) // 2 < total_diffs:
        # can't be below, but we are at best_scope; check if it's the bound
        pass

    def validate(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def scope_of(rows):
        return max(max(r) for r in rows)

    # Backtracking search for a single row given the set of used diffs
    def build_row(used, k, max_val, rng):
        row = [0]
        # We need to pick k more values, strictly increasing, all in (0, max_val]
        # such that all new diffs are not in used
        # Use recursive backtracking with pruning
        def bt(row, used_local):
            if len(row) == k + 1:
                return True
            last = row[-1]
            # Try next values from last+1 up to max_val
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used_local:
                    continue
                row.append(x)
                used_new = used_local | new_diffs
                if bt(row, used_new):
                    return True
                row.pop()
            return False
        if bt(row, used):
            return row
        return None

    # Local search: try to reduce the max by re-optimizing one row at a time
    def local_search(rows, used, rng):
        # Try to reduce scope by re-building rows in random order with tighter max
        improved = True
        while improved and time_left() > 5:
            improved = False
            order = list(range(n_rows))
            rng.shuffle(order)
            for idx in order:
                # Try to rebuild row idx with a smaller max
                current_max = rows[idx][-1]
                # Try to find a row with max < current_max
                for target_max in range(current_max - 1, 0, -1):
                    if time_left() < 5:
                        break
                    # Compute used without this row
                    other_used = set()
                    for j, r in enumerate(rows):
                        if j == idx:
                            continue
                        for i in range(len(r)):
                            for jj in range(i + 1, len(r)):
                                other_used.add(r[jj] - r[i])
                    # Try to build a new row with max <= target_max
                    new_row = build_row(other_used, k, target_max, rng)
                    if new_row is not None:
                        rows[idx] = new_row
                        # Rebuild used
                        used = set()
                        for r in rows:
                            for i in range(len(r)):
                                for jj in range(i + 1, len(r)):
                                    used.add(r[jj] - r[i])
                        improved = True
                        break
        return rows

    # Randomized restarts with greedy + perturbation
    rng = random.Random(42)
    num_restarts = 0
    while time_left() > 10:
        num_restarts += 1
        # Randomized greedy: at each step, among valid candidates, pick randomly from top few
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        
        if not validate(rows):
            continue
        
        sc = scope_of(rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = [r[:] for r in rows]
            # Try local search to improve further
            rows_ls = [r[:] for r in rows]
            used_ls = set()
            for r in rows_ls:
                for i in range(len(r)):
                    for jj in range(i + 1, len(r)):
                        used_ls.add(r[jj] - r[i])
            rows_ls = local_search(rows_ls, used_ls, rng)
            sc_ls = scope_of(rows_ls)
            if sc_ls < best_scope:
                best_scope = sc_ls
                best_rows = [r[:] for r in rows_ls]

        # If we hit the counting bound, we can stop
        if best_scope * (best_scope + 1) // 2 < total_diffs:
            break

    return best_rows
