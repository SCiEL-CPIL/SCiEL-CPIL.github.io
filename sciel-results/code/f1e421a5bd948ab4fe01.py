"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    L = k + 1  # length of each row
    diffs_per_row = k * (k + 1) // 2

    def scope_of(rows):
        return max(max(r) for r in rows)

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def pure_greedy():
        """Fast greedy construction."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def randomized_greedy(window=80):
        """Greedy with shuffled candidate window."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                cands = list(range(x + 1, x + window + 1))
                random.shuffle(cands)
                found = False
                for cand in cands:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= new
                        x = cand
                        found = True
                        break
                if not found:
                    cand = x + 1
                    while True:
                        new = {cand - a for a in row}
                        if not new & used:
                            row.append(cand)
                            used |= new
                            x = cand
                            break
                        cand += 1
            rows.append(row)
        return rows

    def local_search(rows, max_iters, time_limit):
        """
        Local search with multiple move types:
        1. Change a single entry in a row
        2. Swap two entries in a row (if it maintains ordering, this is tricky)
        3. Shift a suffix of a row by a small amount
        
        We focus on single-entry changes and suffix shifts.
        """
        rows = [r[:] for r in rows]
        best = [r[:] for r in rows]
        best_scope = scope_of(rows)
        n = len(rows)
        L = len(rows[0])

        # Precompute per-row diff sets
        row_diff_sets = [row_diffs_set(r) for r in rows]
        all_diffs = set()
        for s in row_diff_sets:
            all_diffs |= s

        T = 100.0
        T_min = 0.1
        cooling = 0.9995

        it = 0
        while it < max_iters and time.time() < time_limit:
            it += 1
            T = max(T_min, T * cooling)

            # Choose a move type
            move_type = random.random()
            ri = random.randrange(n)

            if move_type < 0.7:
                # Single entry change
                ci = random.randrange(1, L)
                old_val = rows[ri][ci]
                old_row = rows[ri]
                old_diffs = row_diff_sets[ri]

                # Generate candidate values
                candidates = set()
                for d in range(-30, 31):
                    nv = old_val + d
                    if nv >= 1:
                        candidates.add(nv)
                for _ in range(15):
                    nv = random.randint(1, best_scope + 50)
                    candidates.add(nv)

                # Enforce ordering
                lower = old_row[ci - 1] + 1
                upper = (old_row[ci + 1] - 1) if ci + 1 < L else best_scope + 50
                candidates = [c for c in candidates if lower <= c <= upper]
                if not candidates:
                    continue

                random.shuffle(candidates)
                for nv in candidates:
                    if nv == old_val:
                        continue
                    new_row = old_row[:]
                    new_row[ci] = nv
                    new_diffs = row_diffs_set(new_row)

                    # Check internal consistency
                    if len(new_diffs) != diffs_per_row:
                        continue

                    # Check no collision with other rows
                    other_diffs = all_diffs - old_diffs
                    if new_diffs & other_diffs:
                        continue

                    # Valid move
                    new_scope = max(best_scope, max(new_row))
                    delta = new_scope - best_scope

                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        # Accept move
                        all_diffs = (all_diffs - old_diffs) | new_diffs
                        row_diff_sets[ri] = new_diffs
                        rows[ri] = new_row

                        if new_scope < best_scope:
                            best = [r[:] for r in rows]
                            best_scope = new_scope
                        break

            else:
                # Suffix shift: pick a row and a column, shift all elements from ci to end by d
                ci = random.randrange(1, L)
                old_row = rows[ri]
                old_diffs = row_diff_sets[ri]
                
                # The shift amount d
                d = random.randint(-20, 20)
                if d == 0:
                    continue
                
                # Check if shifting maintains strict increase
                # new_row[ci:] = old_row[ci:] + d
                # We need old_row[ci-1] < old_row[ci] + d < old_row[ci+1] + d < ...
                # The critical constraint is old_row[ci-1] < old_row[ci] + d
                # i.e., d > old_row[ci-1] - old_row[ci]
                # Since old_row is increasing, old_row[ci-1] - old_row[ci] < 0, so this is usually fine for small d.
                # But we also need new_row[ci] >= 1 if ci == 0 (not the case here)
                
                if ci > 0 and old_row[ci-1] + 1 > old_row[ci] + d:
                    continue
                
                new_row = old_row[:]
                for j in range(ci, L):
                    new_row[j] = old_row[j] + d
                
                # Check all values are positive
                if min(new_row) < 0:
                    continue
                
                # Check strict increase
                for j in range(1, L):
                    if new_row[j] <= new_row[j-1]:
                        break
                else:
                    new_diffs = row_diffs_set(new_row)
                    if len(new_diffs) != diffs_per_row:
                        continue
                    other_diffs = all_diffs - old_diffs
                    if new_diffs & other_diffs:
                        continue
                    
                    new_scope = max(best_scope, max(new_row))
                    delta = new_scope - best_scope
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        all_diffs = (all_diffs - old_diffs) | new_diffs
                        row_diff_sets[ri] = new_diffs
                        rows[ri] = new_row
                        if new_scope < best_scope:
                            best = [r[:] for r in rows]
                            best_scope = new_scope
                        break

        return best, best_scope

    def reduce_scope(rows, time_limit):
        """Try to reduce the scope by moving the largest entries down."""
        rows = [r[:] for r in rows]
        best_scope = scope_of(rows)
        n = len(rows)
        L = len(rows[0])

        row_diff_sets = [row_diffs_set(r) for r in rows]
        all_diffs = set()
        for s in row_diff_sets:
            all_diffs |= s

        # Sort rows by their max value descending, try to reduce the largest first
        for _ in range(200):
            if time.time() > time_limit:
                break
            # Find the row with the current maximum
            max_row_idx = max(range(n), key=lambda i: rows[i][-1])
            row = rows[max_row_idx]

            # Try to reduce the last element
            old_last = row[-1]
            for nv in range(old_last - 1, row[-2], -1):
                new_row = row[:-1] + [nv]
                new_diffs = row_diffs_set(new_row)
                if len(new_diffs) != diffs_per_row:
                    continue
                old_diffs = row_diff_sets[max_row_idx]
                other_diffs = all_diffs - old_diffs
                if new_diffs & other_diffs:
                    continue
                # Valid
                all_diffs = (all_diffs - old_diffs) | new_diffs
                row_diff_sets[max_row_idx] = new_diffs
                rows[max_row_idx] = new_row
                best_scope = scope_of(rows)
                break

        return rows, best_scope

    # ---------- Main search ----------
    # Start with pure greedy
    best_rows = pure_greedy()
    best_scope = scope_of(best_rows)

    # Phase 1: Randomized greedy restarts
    attempts = 0
    while time.time() < deadline - 60.0 and attempts < 300:
        attempts += 1
        rows = randomized_greedy(window=30 + random.randint(0, 100))
        sc = scope_of(rows)
        if sc < best_scope:
            best_rows = rows
            best_scope = sc
            if best_scope <= 112:
                break

    # Phase 2: Local search on best and a few alternatives
    pool = [best_rows]
    for _ in range(10):
        if time.time() >= deadline - 40.0:
            break
        rows = randomized_greedy(window=30 + random.randint(0, 100))
        sc = scope_of(rows)
        if sc <= best_scope + 40:
            pool.append(rows)

    for start in pool:
        if time.time() >= deadline - 20.0:
            break
        improved, sc = local_search(start, max_iters=20000, time_limit=deadline - 20.0)
        if sc < best_scope:
            best_rows = improved
            best_scope = sc
            if best_scope <= 112:
                break

    # Phase 3: Scope reduction pass
    if time.time() < deadline - 5.0:
        reduced, sc = reduce_scope(best_rows, deadline - 5.0)
        if sc < best_scope:
            best_rows = reduced
            best_scope = sc

    return best_rows
