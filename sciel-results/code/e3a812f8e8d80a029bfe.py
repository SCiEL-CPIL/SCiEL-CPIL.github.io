import time
import random
import math
import sys
from collections import Counter

def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Number of differences per row
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    # Helper: compute scope (max entry)
    def _scope(rows):
        return max(a for r in rows for a in r[1:])

    # Helper: fast greedy construction
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                valid = True
                for a in row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows

    # Get baseline
    baseline_rows = greedy_construct()
    best_rows = [r[:] for r in baseline_rows]
    best_max = max(max(r) for r in best_rows)

    # Build diff counts
    def build_diff_counts(rows):
        counts = Counter()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    counts[d] += 1
        return counts

    def remove_row_diffs(diff_counts, row):
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]

    def add_row_diffs(diff_counts, row):
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                diff_counts[d] = diff_counts.get(d, 0) + 1

    def get_cost(rows, diff_counts):
        scope = _scope(rows)
        collisions = sum(max(0, c - 1) for c in diff_counts.values())
        return scope + 500 * collisions

    # CSP-based row construction
    def csp_build_row(k, used_diffs, max_val, rng, time_limit):
        start = time.time()
        best_row = None
        best_score = float('inf')

        for attempt in range(30):
            if time.time() - start > time_limit:
                break

            row = [0]
            local_used = set()
            ok = True

            for j in range(1, k + 1):
                found = False
                min_val = row[-1] + 1
                max_val_candidate = max_val - (k - j)

                if max_val_candidate < min_val:
                    ok = False
                    break

                if attempt < 5:
                    candidates = range(min_val, max_val_candidate + 1)
                else:
                    start_offset = rng.randint(0, max(0, max_val_candidate - min_val))
                    candidates = list(range(min_val + start_offset, max_val_candidate + 1))
                    if min_val + start_offset > min_val:
                        candidates += list(range(min_val, min_val + start_offset))
                    rng.shuffle(candidates)

                for v in candidates:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = v - a
                        if d in used_diffs or d in local_used:
                            valid = False
                            break
                        new_diffs.append(d)

                    if valid:
                        row.append(v)
                        local_used.update(new_diffs)
                        found = True
                        break

                if not found:
                    ok = False
                    break

            if ok:
                score = row[-1]
                if score < best_score:
                    best_score = score
                    best_row = list(row)
                    if best_score <= k * (k + 1) // 2:
                        break

        return best_row

    # Main SA optimization with row-rebuild moves
    def sa_optimize(rows, k, deadline, polish=False):
        rng = random.Random()
        current_rows = [list(r) for r in rows]
        current_scope = _scope(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_scope = current_scope

        diff_counts = build_diff_counts(current_rows)
        current_cost = get_cost(current_rows, diff_counts)
        best_cost = current_cost

        if polish:
            T = 0.5
            cooling = 0.9999
            min_T = 0.01
        else:
            T = 8.0
            cooling = 0.9995
            min_T = 0.05

        while time.time() < deadline:
            r = rng.random()

            if r < 0.3:
                # Row rebuild move
                i = rng.randint(0, len(current_rows) - 1)
                old_row = list(current_rows[i])

                old_diffs = set()
                for a in range(len(old_row)):
                    for b in range(a + 1, len(old_row)):
                        old_diffs.add(old_row[b] - old_row[a])

                remove_row_diffs(diff_counts, old_row)
                used_diffs = set(diff_counts.keys())

                # Try to rebuild with tight scope
                new_row = csp_build_row(k, used_diffs, current_scope, rng, 0.03)
                if new_row is None:
                    new_row = csp_build_row(k, used_diffs, current_scope + 5, rng, 0.03)

                if new_row is None:
                    add_row_diffs(diff_counts, old_row)
                    continue

                new_diffs = set()
                for a in range(len(new_row)):
                    for b in range(a + 1, len(new_row)):
                        new_diffs.add(new_row[b] - new_row[a])

                add_row_diffs(diff_counts, new_row)
                current_rows[i] = new_row

                new_scope = _scope(current_rows)
                new_cost = get_cost(current_rows, diff_counts)

                if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                    current_cost = new_cost
                    current_scope = new_scope
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
                else:
                    # Revert
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    current_rows[i] = old_row
                    current_scope = _scope(current_rows)

            elif r < 0.5:
                # Scope compression: target the max element
                max_val = current_scope
                max_row_idx = -1
                max_pos_idx = -1
                for ri in range(len(current_rows)):
                    for pi in range(k, 0, -1):
                        if current_rows[ri][pi] == max_val:
                            max_row_idx = ri
                            max_pos_idx = pi
                            break
                    if max_row_idx >= 0:
                        break

                if max_row_idx < 0:
                    continue

                i = max_row_idx
                j = max_pos_idx
                old_val = current_rows[i][j]
                prev_val = current_rows[i][j - 1]

                if old_val <= prev_val + 1:
                    continue

                new_val = rng.randint(prev_val + 1, old_val - 1)

                remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = new_val
                add_row_diffs(diff_counts, current_rows[i])

                new_scope = _scope(current_rows)
                new_cost = get_cost(current_rows, diff_counts)

                if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                    current_cost = new_cost
                    current_scope = new_scope
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
                else:
                    # Revert
                    remove_row_diffs(diff_counts, current_rows[i])
                    current_rows[i][j] = old_val
                    add_row_diffs(diff_counts, current_rows[i])
                    current_scope = _scope(current_rows)

            else:
                # Single element move
                r_idx = rng.randint(0, len(current_rows) - 1)
                i_idx = rng.randint(1, row_len - 1)

                lower = current_rows[r_idx][i_idx - 1] + 1
                if i_idx < row_len - 1:
                    upper = current_rows[r_idx][i_idx + 1] - 1
                else:
                    upper = current_scope + 5

                if lower > upper:
                    continue

                new_val = rng.randint(lower, upper)
                old_val = current_rows[r_idx][i_idx]

                if new_val == old_val:
                    continue

                remove_row_diffs(diff_counts, current_rows[r_idx])
                current_rows[r_idx][i_idx] = new_val
                add_row_diffs(diff_counts, current_rows[r_idx])

                new_scope = _scope(current_rows)
                new_cost = get_cost(current_rows, diff_counts)

                if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                    current_cost = new_cost
                    current_scope = new_scope
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
                else:
                    # Revert
                    remove_row_diffs(diff_counts, current_rows[r_idx])
                    current_rows[r_idx][i_idx] = old_val
                    add_row_diffs(diff_counts, current_rows[r_idx])
                    current_scope = _scope(current_rows)

            T *= cooling
            if T < min_T:
                T = min_T

        return best_rows, best_scope

    # Main loop
    while time.time() < deadline:
        # Create initial state
        if random.random() < 0.3:
            # Use best rows as starting point
            init_rows = [r[:] for r in best_rows]
        else:
            # Random initialization within a bound
            bound = best_max + 5
            init_rows = []
            for _ in range(n_rows):
                pts = random.sample(range(1, bound + 1), k)
                pts.sort()
                init_rows.append([0] + pts)

        # Run SA for a fixed time
        sa_time = min(8.0, deadline - time.time())
        if sa_time < 1.0:
            break

        local_rows, local_max = sa_optimize(init_rows, k, deadline)

        if local_max < best_max:
            best_rows = local_rows
            best_max = local_max

    return best_rows
