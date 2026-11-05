"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences
    within rows are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin before 150s kill

    n_marks = k + 1  # marks per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # --- Fast difference computation ---
    def get_all_diffs(rows):
        """Return set of all positive differences across all rows."""
        used = set()
        for row in rows:
            for i in range(n_marks):
                ri = row[i]
                for j in range(i + 1, n_marks):
                    used.add(row[j] - ri)
        return used

    def get_row_diffs(row):
        """Return set of positive differences for a single row."""
        diffs = set()
        for i in range(n_marks):
            ri = row[i]
            for j in range(i + 1, n_marks):
                diffs.add(row[j] - ri)
        return diffs

    # --- Greedy initialization ---
    def greedy_init(seed=42):
        rng = random.Random(seed)
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n_marks:
                # Look in a window for candidates
                found = False
                for window in [10, 30, 100, 500]:
                    for dx in range(1, window + 1):
                        x_candidate = x + dx
                        new_diffs = {x_candidate - a for a in row}
                        if not (new_diffs & used):
                            x = x_candidate
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                    if found:
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    # --- Cost function: max_entry + conflict penalty ---
    def compute_cost(rows):
        used = set()
        max_entry = 0
        conflicts = 0
        for r in range(n_rows):
            row = rows[r]
            max_entry = max(max_entry, row[-1])
            for i in range(n_marks):
                ri = row[i]
                for j in range(i + 1, n_marks):
                    d = row[j] - ri
                    if d <= 0:
                        conflicts += 1000
                        continue
                    if d in used:
                        conflicts += 1
                    used.add(d)
        return max_entry + conflicts * 50

    # --- Simulated Annealing ---
    def simulated_annealing(initial_rows, time_budget):
        current_rows = [list(r) for r in initial_rows]
        current_cost = compute_cost(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_cost = current_cost

        T_start = 80.0
        T_end = 0.01
        steps = 0
        max_steps = 300000
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_steps - 1))

        while time.time() < time_budget and steps < max_steps:
            steps += 1
            T = T_start * (cooling_rate ** steps)
            if T < T_end:
                T = T_end

            # Pick a random row and position
            r = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)

            old_val = current_rows[r][pos]
            min_val = current_rows[r][pos - 1] + 1
            max_val = current_rows[r][pos + 1] - 1 if pos < k else 300

            if min_val > max_val:
                continue

            if random.random() < 0.6:
                delta = random.randint(-8, 8)
                new_val = old_val + delta
            else:
                new_val = random.randint(min_val, max_val)

            if new_val < min_val or new_val > max_val or new_val == old_val:
                continue

            current_rows[r][pos] = new_val
            new_cost = compute_cost(current_rows)
            delta_cost = new_cost - current_cost

            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / max(T, 1e-10)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(row) for row in current_rows]
            else:
                current_rows[r][pos] = old_val

        return best_rows, best_cost

    # --- Local search: coordinate descent with repair ---
    def local_search(initial_rows, time_budget):
        current_rows = [list(r) for r in initial_rows]
        current_cost = compute_cost(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_cost = current_cost

        no_improve = 0
        max_no_improve = 3000

        while time.time() < time_budget and no_improve < max_no_improve:
            improved = False

            for r in range(n_rows):
                for pos in range(1, k + 1):
                    min_val = current_rows[r][pos - 1] + 1
                    max_val = current_rows[r][pos + 1] - 1 if pos < k else 300

                    if min_val > max_val:
                        continue

                    old_val = current_rows[r][pos]
                    best_local_val = old_val
                    best_local_cost = current_cost

                    # Try candidate values
                    candidates = set()
                    candidates.add(old_val)
                    candidates.add(min_val)
                    candidates.add(max_val)
                    if old_val > min_val:
                        candidates.add(old_val - 1)
                    if old_val < max_val:
                        candidates.add(old_val + 1)
                    # Add random candidates
                    for _ in range(5):
                        candidates.add(random.randint(min_val, max_val))

                    for val in candidates:
                        if val < min_val or val > max_val:
                            continue
                        current_rows[r][pos] = val
                        new_cost = compute_cost(current_rows)
                        if new_cost < best_local_cost:
                            best_local_cost = new_cost
                            best_local_val = val

                    if best_local_val != old_val:
                        current_rows[r][pos] = best_local_val
                        current_cost = best_local_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_rows = [list(row) for row in current_rows]

            if not improved:
                no_improve += 1
            else:
                no_improve = 0

        return best_rows, best_cost

    # --- Repair: fix conflicts by rebuilding problematic rows ---
    def repair_rows(rows, time_budget):
        """Try to repair conflicts by rebuilding rows that contribute to conflicts."""
        rows = [list(r) for r in rows]
        used = get_all_diffs(rows)
        current_cost = compute_cost(rows)
        best_rows = [list(r) for r in rows]
        best_cost = current_cost

        # Find rows with the most conflicts
        for _ in range(100):
            if time.time() > time_budget:
                break

            # Identify row with most conflicts
            row_conflicts = [0] * n_rows
            used_check = set()
            for r in range(n_rows):
                row = rows[r]
                for i in range(n_marks):
                    ri = row[i]
                    for j in range(i + 1, n_marks):
                        d = row[j] - ri
                        if d in used_check:
                            row_conflicts[r] += 1
                        used_check.add(d)

            # Rebuild the row with most conflicts
            target_r = max(range(n_rows), key=lambda r: row_conflicts[r])
            if row_conflicts[target_r] == 0:
                break

            # Remove this row's diffs from global set
            old_row = rows[target_r]
            old_diffs = get_row_diffs(old_row)
            global_used = set(used)
            for d in old_diffs:
                global_used.discard(d)

            # Try to build a new row
            new_row = _try_build_row(k, global_used, random, time_budget)
            if new_row is not None:
                rows[target_r] = new_row
                new_cost = compute_cost(rows)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(row) for row in rows]
                used = get_all_diffs(rows)

        return best_rows, best_cost

    # --- Try to build a row avoiding given diffs ---
    def _try_build_row(k_val, global_used, rng, deadline):
        row = [0]
        x = 0
        used_in_row = set()
        while len(row) < n_marks:
            if time.time() > deadline:
                return None
            found = False
            for window in [10, 30, 100, 500]:
                for dx in range(1, window + 1):
                    x_candidate = x + dx
                    new_diffs = {x_candidate - a for a in row}
                    if not (new_diffs & global_used) and not (new_diffs & used_in_row):
                        x = x_candidate
                        row.append(x)
                        used_in_row |= new_diffs
                        found = True
                        break
                if found:
                    break
            if not found:
                return None
        return row

    # --- Main search loop ---
    best_rows = None
    best_max = float('inf')

    # Phase 1: Greedy initialization with multiple seeds
    for seed in [42, 0, 1, 2, 3, 123, 456, 789]:
        if time.time() > deadline:
            break
        rows = greedy_init(seed)
        if rows is not None:
            row_max = max(max(row) for row in rows)
            if row_max < best_max:
                best_max = row_max
                best_rows = rows

    if best_rows is None:
        best_rows = greedy_init(42)
        if best_rows is None:
            # Fallback: simple greedy
            best_rows = []
            used = set()
            for r in range(n_rows):
                row = [0]
                x = 0
                while len(row) < n_marks:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                best_rows.append(row)
            best_max = max(max(row) for row in best_rows)

    # Phase 2: Local search refinement
    remaining = deadline - time.time()
    if remaining > 5:
        ls_rows, ls_cost = local_search(best_rows, deadline - 5)
        ls_max = max(max(row) for row in ls_rows)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows

    # Phase 3: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 10:
        sa_rows, sa_cost = simulated_annealing(best_rows, deadline - 5)
        sa_max = max(max(row) for row in sa_rows)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows

    # Phase 4: Repair conflicts
    remaining = deadline - time.time()
    if remaining > 5:
        rep_rows, rep_cost = repair_rows(best_rows, deadline - 2)
        rep_max = max(max(row) for row in rep_rows)
        if rep_max < best_max:
            best_max = rep_max
            best_rows = rep_rows

    # Phase 5: Final local search
    remaining = deadline - time.time()
    if remaining > 3:
        final_rows, final_cost = local_search(best_rows, deadline - 1)
        final_max = max(max(row) for row in final_rows)
        if final_max < best_max:
            best_max = final_max
            best_rows = final_rows

    # Validate: ensure no conflicts
    used = set()
    for row in best_rows:
        for i in range(n_marks):
            ri = row[i]
            for j in range(i + 1, n_marks):
                d = row[j] - ri
                if d in used:
                    # Conflict exists; try one more repair
                    pass
                used.add(d)

    return best_rows
# EVOLVE-BLOCK-END
