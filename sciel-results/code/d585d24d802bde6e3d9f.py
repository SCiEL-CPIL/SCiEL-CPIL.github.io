"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 145.0
    rng = random.Random(123456789)

    # Precompute the list of (i, j) index-pairs within a row, i < j, in a
    # deterministic order. The j-th element of a row is a[j] = a[0] + sum of
    # differences a[j]-a[i] over i in 0..j-1, so we can reconstruct a row
    # entirely from the difference values assigned to these slots.
    pairs = [(i, j) for j in range(k + 1) for i in range(j)]  # order: by j
    # For each position j (1..k), the differences that define a[j] relative to
    # earlier marks are (i, j) for i in 0..j-1.
    # We build a row incrementally: at step j we know a[0..j-1], and the
    # difference a[j]-a[i] for i<j are the slots that must be assigned.

    def diffs_of_row(row):
        return [row[j] - row[i] for (i, j) in pairs]

    def all_diffs(rows):
        out = []
        for row in rows:
            out.extend(diffs_of_row(row))
        return out

    def max_entry(rows):
        return max(max(row) for row in rows)

    def count_violations(rows):
        diffs = all_diffs(rows)
        counts = {}
        for d in diffs:
            counts[d] = counts.get(d, 0) + 1
        return sum(c - 1 for c in counts.values() if c > 1)

    def compute_cost(rows):
        return count_violations(rows) * 100000 + max_entry(rows)

    # --- Difference-assignment greedy: build one row by assigning the
    #     smallest free difference to each slot, preferring assignments that
    #     keep the running mark consistent and small. This is the
    #     "graph coloring / smallest-available-label" structural idea: the
    #     difference slots are colored (assigned values) so no two slots in
    #     the whole solution share a value, and we greedily give each slot the
    #     smallest unused value that is consistent with the partial row. ---
    def assign_row_greedy(used, jitter=0.0, rng_=rng):
        row = [0]
        # At step j we have row[0..j-1]. The new mark a[j] must satisfy
        # a[j] > a[j-1]. The differences a[j]-a[i] (i<j) must all be unused.
        # We try candidate marks a[j] in increasing order (optionally with a
        # small random perturbation for diversity) and pick the first whose
        # full set of differences is free.
        for j in range(1, k + 1):
            prev = row[j - 1]
            x = prev
            found = False
            # Number of candidate marks to try before giving up on this j.
            # Bias: smaller marks first to keep scope small.
            tries = 0
            max_tries = 200000
            while not found and tries < max_tries:
                x += 1
                new = [x - row[i] for i in range(j)]
                ok = True
                for d in new:
                    if d in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for d in new:
                        used.add(d)
                    found = True
                    break
                tries += 1
            if not found:
                return None
        return row

    def greedy_init(seed_rng):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = assign_row_greedy(used, rng_=seed_rng)
            if row is None:
                return None
            rows.append(row)
        return rows

    best_rows = None
    best_max = None
    # Multi-start: try several randomized orderings / seeds to get diverse
    # starting points (borrowed from p47's randomized_greedy idea).
    for s in range(6):
        if time.time() > deadline - 5:
            break
        sr = random.Random(1000 + s)
        rows = greedy_init(sr)
        if rows is None:
            continue
        m = max_entry(rows)
        if best_max is None or m < best_max:
            best_max = m
            best_rows = [r[:] for r in rows]
    if best_rows is None:
        # Fallback: pure deterministic greedy (the seed behavior).
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = assign_row_greedy(used, rng_=rng)
            best_rows.append(row)
        best_max = max_entry(best_rows)

    # --- Simulated-annealing refinement on the marks (from inspiration 1),
    #     with incremental diff-count bookkeeping for speed. ---
    current_rows = [r[:] for r in best_rows]
    current_cost = compute_cost(current_rows)

    # Incremental diff counts: map difference value -> count.
    diff_count = {}
    for row in current_rows:
        for d in diffs_of_row(row):
            diff_count[d] = diff_count.get(d, 0) + 1
    violations = sum(c - 1 for c in diff_count.values() if c > 1)

    def row_diffs(row):
        return [row[j] - row[i] for (i, j) in pairs]

    def apply_move(r, i, new_val):
        """Change current_rows[r][i] to new_val, updating diff_count and
        violations incrementally. Returns (old_val, delta_violations)."""
        row = current_rows[r]
        old_val = row[i]
        if new_val == old_val:
            return old_val, 0
        # Remove old differences involving position i.
        for (a, b) in pairs:
            if a == i or b == i:
                d = row[b] - row[a]
                diff_count[d] -= 1
                if diff_count[d] == 1:
                    violations -= 1
                elif diff_count[d] == 0:
                    del diff_count[d]
        # Set new value.
        row[i] = new_val
        # Add new differences involving position i.
        for (a, b) in pairs:
            if a == i or b == i:
                d = row[b] - row[a]
                if d in diff_count:
                    if diff_count[d] == 1:
                        violations += 1
                    diff_count[d] += 1
                else:
                    diff_count[d] = 1
        return old_val, 0

    # We recompute violations from scratch occasionally to avoid drift.
    def recompute_violations():
        nonlocal violations
        violations = sum(c - 1 for c in diff_count.values() if c > 1)

    indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]

    T = 40.0
    T_min = 0.05
    cooling = 0.99997
    step = 0

    while time.time() < deadline - 3.0:
        if T < T_min:
            T = T_min  # keep a little random exploration going

        r, i = indices[rng.randrange(len(indices))]
        row = current_rows[r]
        lower = row[i - 1] + 1
        upper = row[i + 1] - 1 if i < k else row[k] + 40
        if upper < lower:
            continue

        # Bias toward shrinking to reduce max entry.
        if rng.random() < 0.35:
            span = upper - lower + 1
            new_val = lower + int(rng.random() * span * 0.5)
        else:
            new_val = rng.randint(lower, upper)

        old_val = row[i]
        if new_val == old_val:
            continue

        # Apply, checking strict-increasing (guaranteed by range).
        # Compute cost change via incremental violations + max change.
        old_max = max_entry(current_rows)
        # Temporarily apply
        row[i] = new_val
        # recompute diffs for this row only for diff_count
        # Remove old
        for (a, b) in pairs:
            if a == i or b == i:
                d = old_val - row[a] if b == i else row[b] - old_val
                diff_count[d] -= 1
                if diff_count[d] == 1:
                    violations -= 1
                elif diff_count[d] == 0:
                    del diff_count[d]
        # Add new
        for (a, b) in pairs:
            if a == i or b == i:
                d = new_val - row[a] if b == i else row[b] - new_val
                if d in diff_count:
                    if diff_count[d] == 1:
                        violations += 1
                    diff_count[d] += 1
                else:
                    diff_count[d] = 1
        new_max = max(row) if max(row) > max(max(rr) for rr in current_rows if rr is not row) else max(max(rr) for rr in current_rows)
        # Simpler: new_max over all rows
        new_max = 0
        for rr in current_rows:
            if rr[-1] > new_max:
                new_max = rr[-1]
        old_cost = violations * 100000 + old_max
        new_cost = violations * 100000 + new_max
        delta = new_cost - old_cost
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
            # accept
            if violations == 0 and new_max < best_max:
                best_max = new_max
                best_rows = [rr[:] for rr in current_rows]
        else:
            # reject: revert
            row[i] = old_val
            for (a, b) in pairs:
                if a == i or b == i:
                    d = new_val - row[a] if b == i else row[b] - new_val
                    diff_count[d] -= 1
                    if diff_count[d] == 1:
                        violations -= 1
                    elif diff_count[d] == 0:
                        del diff_count[d]
            for (a, b) in pairs:
                if a == i or b == i:
                    d = old_val - row[a] if b == i else row[b] - old_val
                    if d in diff_count:
                        if diff_count[d] == 1:
                            violations += 1
                        diff_count[d] += 1
                    else:
                        diff_count[d] = 1
        T *= cooling
        step += 1
        if step % 2000 == 0:
            recompute_violations()

    # Final safety: ensure returned rows are valid; if not, fall back to a
    # fresh greedy build (which is guaranteed valid).
    if count_violations(best_rows) != 0:
        used = set()
        fb = []
        for _ in range(n_rows):
            row = assign_row_greedy(used, rng_=rng)
            fb.append(row)
        if max_entry(fb) <= best_max:
            best_rows = fb
        else:
            # keep best_rows only if valid; otherwise use fb
            best_rows = fb

    return best_rows
# EVOLVE-BLOCK-END
