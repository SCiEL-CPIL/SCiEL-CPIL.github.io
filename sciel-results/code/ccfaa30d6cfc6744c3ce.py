"""Difference triangle set solver using Iterated Local Search with kick perturbations."""
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Per-row local optimization (refinement machinery) ----------
    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def try_reduce_row(rows, row_idx, used_other, max_attempts=200):
        """Try to find a row at row_idx with smaller max entry (same k+1 size)."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 3:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + 200):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) < best_max:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        while improved and time_left() > 12:
            improved = False
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 12:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=250)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(best_rows)

    # ---------- Simulated Annealing (exploration machinery) ----------
    # Conflict counting via integer array for fast delta evaluation.
    if time_left() > 40:
        n_diffs_per_row = k * (k + 1) // 2  # number of differences per row
        # Order rows by descending scope: index 0 is the binding constraint.
        order = sorted(range(n_rows), key=lambda i: -max(best_rows[i]))
        sa_rows = [list(best_rows[i]) for i in order]

        # Global conflict count array.
        max_diff = max(max(r) for r in sa_rows) + 1
        conflict = [0] * (max_diff + 1)
        for row in sa_rows:
            for d in row_diffs_list(row):
                if d > max_diff:
                    max_diff = d
                    conflict = [0] * (max_diff + 1)
                conflict[d] += 1

        # Track number of conflicts incrementally
        cur_conflict_count = sum(1 for c in conflict if c > 1)

        def total_conflicts():
            nonlocal cur_conflict_count
            return cur_conflict_count

        def update_conflict(d, delta):
            """Update conflict count when adding/removing a difference d."""
            nonlocal cur_conflict_count
            old = conflict[d]
            new = old + delta
            if old == 1 and new == 2:
                cur_conflict_count += 1
            elif old == 2 and new == 1:
                cur_conflict_count -= 1
            conflict[d] = new

        def row_diffs_counts(row):
            """Return (diffs list, internal duplicate count) for a row."""
            counts = {}
            dup = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in counts:
                        dup += 1
                    counts[d] = counts.get(d, 0) + 1
            return list(counts.keys()), dup

        def remove_row_diffs(row):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    update_conflict(d, -1)

        def add_row_diffs(row):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d >= len(conflict):
                        conflict.extend([0] * (d + 1 - len(conflict)))
                    update_conflict(d, 1)

        def energy():
            # Primary: total conflicts. Secondary: scope of the largest row (index 0).
            return (total_conflicts(), max(sa_rows[0]))

        cur_conf, cur_scope = energy()
        best_sa = [list(r) for r in sa_rows]
        best_energy = (cur_conf, cur_scope)

        # Precompute per-row differences for fast delta.
        row_diffs_cache = [row_diffs_counts(r)[0] for r in sa_rows]

        def greedy_row_for(used_set):
            """Greedy row given a set of already-used differences (from other rows)."""
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used_set) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
            return row

        def random_row_for(used_set, jitter=0.05):
            """Greedy row with occasional skips to explore alternatives."""
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used_set) and not (new & row_diffs):
                    if jitter and random.random() < jitter and x < 300:
                        continue
                    row.append(x)
                    row_diffs |= new
            return row

        # SA loop: pick a row, remove it, propose a new row, evaluate delta.
        t = 1.0
        t_min = 0.02
        cooling = 0.995
        step = 0
        while time_left() > 20 and t > t_min:
            step += 1
            # Pick a row: bias toward the largest (index 0) early, random later.
            if random.random() < 0.5:
                idx = 0
            else:
                idx = random.randrange(n_rows)

            old_row = sa_rows[idx]
            old_diffs = row_diffs_cache[idx]

            # Remove old row's differences.
            remove_row_diffs(old_row)
            conf_after_remove = total_conflicts()

            # Build used set from remaining rows.
            used_set = set()
            for j in range(n_rows):
                if j != idx:
                    used_set.update(row_diffs_cache[j])

            # Propose new row.
            if random.random() < 0.6:
                new_row = greedy_row_for(used_set)
            else:
                new_row = random_row_for(used_set, jitter=0.1)

            # Ensure new row has exactly k+1 entries (greedy always does).
            if len(new_row) != k + 1:
                # Fallback: revert.
                add_row_diffs(old_row)
                t *= cooling
                continue

            # Add new row's differences, compute new conflict count.
            new_diffs = row_diffs_counts(new_row)[0]
            add_row_diffs(new_row)
            conf_after_add = total_conflicts()

            new_scope = max(new_row) if idx == 0 else max(sa_rows[0])
            # Energy delta: (conflict change, scope change)
            conf_delta = conf_after_add - conf_after_remove
            scope_delta = new_scope - cur_scope

            accept = False
            if conf_delta < 0 or (conf_delta == 0 and scope_delta < 0):
                accept = True
            else:
                # Metropolis on combined penalty.
                penalty = conf_delta * 1000 + scope_delta
                if penalty < 0:
                    accept = True
                else:
                    if random.random() < pow(2.718281828, -penalty / max(t, 1e-6)):
                        accept = True

            if accept:
                sa_rows[idx] = new_row
                row_diffs_cache[idx] = new_diffs
                cur_conf = conf_after_add
                cur_scope = new_scope
                cur_energy = (cur_conf, cur_scope)
                if cur_energy < best_energy:
                    best_energy = cur_energy
                    best_sa = [list(r) for r in sa_rows]
            else:
                # Revert.
                remove_row_diffs(new_row)
                add_row_diffs(old_row)

            t *= cooling

        # Reorder best_sa back to original row order.
        if best_energy[0] == 0:  # valid
            # Map back: best_sa[i] corresponds to original row order[i].
            restored = [None] * n_rows
            for i in range(n_rows):
                restored[order[i]] = best_sa[i]
            if is_valid(restored):
                s = scope(restored)
                if s < best_scope:
                    best_rows = restored
                    best_scope = s

        # Final local polish after SA.
        if time_left() > 12:
            polished = local_polish(best_rows)
            if scope(polished) < best_scope:
                best_rows = polished
                best_scope = scope(best_rows)

    # Final validity guarantee.
    if not is_valid(best_rows):
        # Fallback to a fresh greedy (always valid).
        best_rows = greedy()

    return best_rows
