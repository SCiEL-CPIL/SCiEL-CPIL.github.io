"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
from collections import defaultdict


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # ---------- helpers ----------
    def scope_of(rows):
        return max(max(r) for r in rows)

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def build_global_used(rows):
        """Build a set of all differences across all rows."""
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    used.add(row[j] - row[i])
        return used

    def check_valid(rows):
        """Check if all differences across all rows are distinct."""
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

    def pure_greedy():
        """Pure greedy: pick smallest valid value at each step."""
        used = set()
        rows = []
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
        return rows

    def randomized_greedy(window=80):
        """Greedy with shuffled candidate window for diversity."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
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
        """Local search: perturb single entries, accept if scope decreases."""
        rows = [r[:] for r in rows]
        best = [r[:] for r in rows]
        best_scope = scope_of(rows)
        n = len(rows)
        L = len(rows[0])

        # Precompute used set for fast collision checking
        used = build_global_used(rows)

        it = 0
        while it < max_iters and time.time() < time_limit:
            it += 1
            ri = random.randrange(n)
            if L <= 1:
                break
            ci = random.randrange(1, L)

            old_val = rows[ri][ci]
            old_row = rows[ri]

            # Generate candidates
            candidates = set()
            for d in range(-10, 11):
                nv = old_val + d
                if nv >= 1:
                    candidates.add(nv)
            for _ in range(6):
                nv = random.randint(1, best_scope)
                candidates.add(nv)

            # Enforce strict increasing order
            lower = old_row[ci - 1] + 1
            upper = (old_row[ci + 1] - 1) if ci + 1 < L else best_scope
            candidates = [c for c in candidates if lower <= c <= upper]
            if not candidates:
                continue

            random.shuffle(candidates)
            for nv in candidates:
                if nv == old_val:
                    continue

                # Compute old diffs for this row
                old_diffs = row_diffs_set(old_row)
                # Compute new diffs for this row
                new_row = old_row[:]
                new_row[ci] = nv
                new_diffs = row_diffs_set(new_row)

                # Check internal uniqueness
                if len(new_diffs) != diffs_per_row:
                    continue

                # Check for collisions with global used set (excluding old row's diffs)
                # The global used set contains all diffs from all rows.
                # We need: new_diffs must not intersect (used - old_diffs)
                other_diffs = used - old_diffs
                if new_diffs & other_diffs:
                    continue

                # Valid move! Check if it reduces scope
                new_scope = max(max(r) for i, r in enumerate(rows) if i != ri)
                new_scope = max(new_scope, max(new_row))
                if new_scope < best_scope:
                    # Update used set
                    used -= old_diffs
                    used |= new_diffs
                    rows[ri] = new_row
                    best = [r[:] for r in rows]
                    best_scope = new_scope
                break

        return best, best_scope

    def hill_climb_scope(rows, time_limit):
        """Aggressively try to reduce the maximum entry by targeting the row(s)
        that contain the maximum value."""
        rows = [r[:] for r in rows]
        best = [r[:] for r in rows]
        best_scope = scope_of(rows)
        n = len(rows)
        L = len(rows[0])
        used = build_global_used(rows)

        while time.time() < time_limit:
            # Find row(s) with max entry
            max_val = max(max(r) for r in rows)
            target_rows = [i for i in range(n) if max(rows[i]) == max_val]
            if not target_rows:
                break

            ri = random.choice(target_rows)
            # Try to reduce the last element (or a high-index element)
            ci = L - 1
            old_val = rows[ri][ci]

            # Try to find a smaller value that works
            lower = rows[ri][ci - 1] + 1
            found = False
            for nv in range(lower, old_val):
                new_row = rows[ri][:]
                new_row[ci] = nv
                new_diffs = row_diffs_set(new_row)
                old_diffs = row_diffs_set(rows[ri])
                other_diffs = used - old_diffs
                if new_diffs & other_diffs:
                    continue
                if len(new_diffs) != diffs_per_row:
                    continue
                # Valid
                used -= old_diffs
                used |= new_diffs
                rows[ri] = new_row
                best = [r[:] for r in rows]
                best_scope = scope_of(rows)
                found = True
                break

            if not found:
                # Try other positions in this row
                for ci2 in range(L - 1, 0, -1):
                    if time.time() >= time_limit:
                        break
                    old_val = rows[ri][ci2]
                    lower = rows[ri][ci2 - 1] + 1
                    upper = (rows[ri][ci2 + 1] - 1) if ci2 + 1 < L else best_scope
                    for nv in range(lower, old_val):
                        new_row = rows[ri][:]
                        new_row[ci2] = nv
                        new_diffs = row_diffs_set(new_row)
                        old_diffs = row_diffs_set(rows[ri])
                        other_diffs = used - old_diffs
                        if new_diffs & other_diffs:
                            continue
                        if len(new_diffs) != diffs_per_row:
                            continue
                        # Check if this reduces scope
                        new_scope = max(max(r) for i, r in enumerate(rows) if i != ri)
                        new_scope = max(new_scope, max(new_row))
                        if new_scope < best_scope:
                            used -= old_diffs
                            used |= new_diffs
                            rows[ri] = new_row
                            best = [r[:] for r in rows]
                            best_scope = new_scope
                            break
                    if best_scope < max_val:
                        break

            if best_scope <= 112:
                break

        return best, best_scope

    # ---------- main search ----------
    # Baseline: pure greedy
    base_rows = pure_greedy()
    best_rows = base_rows
    best_scope = scope_of(base_rows)

    # Phase 1: randomized greedy restarts
    attempts = 0
    while time.time() < deadline - 40.0 and attempts < 100:
        attempts += 1
        rows = randomized_greedy(window=random.randint(40, 120))
        sc = scope_of(rows)
        if sc < best_scope:
            best_rows = rows
            best_scope = sc
            if best_scope <= 112:
                break

    # Phase 2: local search on best and a few alternatives
    pool = [best_rows]
    for _ in range(5):
        if time.time() >= deadline - 25.0:
            break
        rows = randomized_greedy(window=random.randint(40, 100))
        sc = scope_of(rows)
        if sc <= best_scope + 20:
            pool.append(rows)

    for start in pool:
        if time.time() >= deadline - 15.0:
            break
        improved, sc = local_search(start, max_iters=8000, time_limit=deadline - 15.0)
        if sc < best_scope:
            best_rows = improved
            best_scope = sc
            if best_scope <= 112:
                break

    # Phase 3: aggressive scope reduction
    if time.time() < deadline - 5.0:
        improved, sc = hill_climb_scope(best_rows, time_limit=deadline - 5.0)
        if sc < best_scope:
            best_rows = improved
            best_scope = sc

    return best_rows
