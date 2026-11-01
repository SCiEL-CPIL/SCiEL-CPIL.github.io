"""Combined program: targeted descent on the global maximum (inspiration 1's structural idea)
with incremental diff-set maintenance and randomized greedy restarts (current program's machinery).
This shifts the search from random point mutations to a directed, greedy reduction of the scope,
which is the actual objective, while using fast incremental validity checks."""
import time
import random

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 145.0

    row_len = k + 1
    d = k * (k + 1) // 2  # diffs per row
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    if diff in seen:
                        return None
                    seen.add(diff)
        return max(max(r) for r in rows)

    def greedy_init(rng, rand_prob=0.15):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                x = row[-1] + 1
                found = False
                while not found:
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                        found = True
                    else:
                        x += 1
            rows.append(row)
        return rows

    def targeted_descent(rows, time_limit, rng):
        """
        Repeatedly find the global maximum element and try to reduce it.
        If stuck, perform a random perturbation to escape.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        cur_rows = [r[:] for r in rows]
        cur_scope = best_scope

        # Pre-compute diff set for current state
        def compute_diff_set(rows):
            s = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        s.add(row[j] - row[i])
            return s

        cur_diff_set = compute_diff_set(cur_rows)

        while time.time() < start + time_limit:
            # Find global max
            max_val = -1
            max_r = -1
            max_c = -1
            for ri, r in enumerate(cur_rows):
                for ci, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r = ri
                        max_c = ci

            if max_c == 0:
                break

            row = cur_rows[max_r]
            prev_val = row[max_c - 1]

            # Compute diffs to remove (those involving position max_c)
            remove_set = set()
            for i in range(max_c):
                remove_set.add(max_val - row[i])
            for j in range(max_c + 1, len(row)):
                remove_set.add(row[j] - max_val)

            fixed_diffs = cur_diff_set - remove_set

            found = False
            # Try to reduce max_val
            for new_val in range(max_val - 1, prev_val, -1):
                valid = True
                new_diffs = set()
                for i in range(max_c):
                    d = new_val - row[i]
                    if d in fixed_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                for j in range(max_c + 1, len(row)):
                    d = row[j] - new_val
                    if d in fixed_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue

                # Valid move
                row[max_c] = new_val
                cur_diff_set = fixed_diffs | new_diffs
                new_scope = get_scope(cur_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur_rows]
                cur_scope = new_scope
                found = True
                break

            if not found:
                # Perturb: pick a random non-max element and move it
                ri = rng.randrange(n_rows)
                ci = rng.randint(1, row_len - 1)
                if ri == max_r and ci == max_c:
                    continue
                old_val = cur_rows[ri][ci]
                lo = cur_rows[ri][ci - 1] + 1
                hi = cur_rows[ri][ci + 1] - 1 if ci < row_len - 1 else best_scope + 20
                if hi <= lo:
                    continue
                new_val = rng.randint(lo, hi)
                if new_val == old_val:
                    continue

                # Compute remove set for this position
                remove_set2 = set()
                r2 = cur_rows[ri]
                for i in range(ci):
                    remove_set2.add(old_val - r2[i])
                for j in range(ci + 1, len(r2)):
                    remove_set2.add(r2[j] - old_val)
                fixed2 = cur_diff_set - remove_set2

                valid2 = True
                new_diffs2 = set()
                for i in range(ci):
                    d = new_val - r2[i]
                    if d in fixed2 or d in new_diffs2:
                        valid2 = False
                        break
                    new_diffs2.add(d)
                if not valid2:
                    continue
                for j in range(ci + 1, len(r2)):
                    d = r2[j] - new_val
                    if d in fixed2 or d in new_diffs2:
                        valid2 = False
                        break
                    new_diffs2.add(d)
                if not valid2:
                    continue

                r2[ci] = new_val
                cur_diff_set = fixed2 | new_diffs2
                new_scope2 = get_scope(cur_rows)
                if new_scope2 < best_scope:
                    best_scope = new_scope2
                    best_rows = [r[:] for r in cur_rows]
                cur_scope = new_scope2

        return best_rows, best_scope

    rng = random.Random(42)
    best_rows = None
    best_scope = float('inf')

    # Multiple restarts with targeted descent
    n_restarts = 50
    per_restart_time = 2.5

    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            rows = greedy_init(rng)
        else:
            # Perturb best
            rows = [r[:] for r in best_rows]
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                ci = rng.randint(1, row_len - 1)
                old = rows[ri][ci]
                lo = rows[ri][ci - 1] + 1
                hi = rows[ri][ci + 1] - 1 if ci < row_len - 1 else best_scope + 10
                if hi > lo:
                    rows[ri][ci] = rng.randint(lo, hi)

        scope = full_validate_scope(rows)
        if scope is None:
            continue

        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

        # Run targeted descent
        t_start = time.time()
        rows_out, scope_out = targeted_descent(rows, per_restart_time, rng)
        if scope_out < best_scope:
            best_scope = scope_out
            best_rows = rows_out

    if best_rows is None:
        best_rows = greedy_init(rng)

    return best_rows
