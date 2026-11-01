"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

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

    def try_reduce_row(rows, row_idx, used_other, max_attempts=100):
        """Try to find a row at row_idx with smaller max entry (same k+1 size)."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            # Randomized greedy for this row to escape local minima in row construction
            jitter = 0.1 if attempt > 10 else 0.0
            while len(row) < k + 1:
                found = False
                # Look ahead a bit
                for trial_x in range(x + 1, x + 50):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        # Apply jitter: sometimes skip a valid candidate to explore
                        if jitter > 0 and random.random() < jitter and trial_x < 200:
                            continue
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
        # Limit the number of full passes to avoid spending too much time here
        # if we are in an ILS loop
        max_passes = 5 
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=50)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def kick(rows):
        """Perturb the solution to escape local minima.
        Targets the row(s) with the largest entries for reconstruction."""
        rows = [list(r) for r in rows]
        # Sort rows by max entry descending; kick the top 1-2
        order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
        num_kicks = random.choice([1, 2])
        indices = order[:num_kicks]
        
        for idx in indices:
            # Calculate used diffs from other rows
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            # Rebuild the row with moderate jitter
            success = False
            for attempt in range(30):
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                jitter = 0.3 if attempt < 10 else 0.15
                while len(row) < k + 1:
                    found = False
                    for trial_x in range(x + 1, x + 80):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            if random.random() < jitter and trial_x < 200:
                                continue
                            row.append(trial_x)
                            current_row_diffs |= new
                            found = True
                            break
                    if not found:
                        valid_row = False
                        break
                    x = row[-1]
                
                if valid_row:
                    rows[idx] = row
                    success = True
                    break
            
            if not success:
                # Fallback: greedy rebuild
                greedy_row, x = [0], 0
                row_diffs = set()
                while len(greedy_row) < k + 1:
                    x += 1
                    new = {x - a for a in greedy_row}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(x)
                        row_diffs |= new
                rows[idx] = greedy_row

        return rows

    # ---------- Simulated Annealing polish ----------
    def sa_polish_row(row, used_other, max_iters=500):
        """SA to minimize sum of squared entries while avoiding used_other diffs."""
        row = list(row)
        n = len(row)
        
        def row_cost(r):
            diffs = set()
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used_other:
                        return float('inf')
                    if d in diffs:
                        return float('inf')
                    diffs.add(d)
            return sum(x * x for x in r)
        
        current_cost = row_cost(row)
        if current_cost == float('inf'):
            return row
        best_row = list(row)
        best_cost = current_cost
        T = 50.0
        cool = 0.995
        
        for it in range(max_iters):
            if time_left() < 3:
                break
            # Pick a random position to modify
            pos = random.randint(1, n - 1)
            old_val = row[pos]
            # Propose a new value
            delta = random.randint(-15, 15)
            new_val = old_val + delta
            # Must maintain strict increase
            if pos > 0 and new_val <= row[pos - 1]:
                new_val = row[pos - 1] + 1
            if pos < n - 1 and new_val >= row[pos + 1]:
                new_val = row[pos + 1] - 1
            if new_val <= row[pos - 1] if pos > 0 else new_val < 0:
                continue
            if pos < n - 1 and new_val >= row[pos + 1]:
                continue
            if new_val == old_val:
                continue
            
            row[pos] = new_val
            new_cost = row_cost(row)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_row = list(row)
            else:
                # Accept with probability exp(-delta/T)
                delta_cost = new_cost - current_cost
                if delta_cost < float('inf') and random.random() < math.exp(-delta_cost / max(T, 0.01)):
                    current_cost = new_cost
                else:
                    row[pos] = old_val
            
            T *= cool
        
        return best_row

    def sa_polish(rows):
        """Apply SA to each row, ordered by scope."""
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 3
        pass_count = 0
        while improved and time_left() > 8 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row = sa_polish_row(rows[i], used_other, max_iters=400)
                if max(new_row) < max(rows[i]):
                    test_rows = [list(r) for r in rows]
                    test_rows[i] = new_row
                    if is_valid(test_rows):
                        rows[i] = new_row
                        improved = True
        return rows

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        if time_left() > 15:
            sa_result = sa_polish(best_rows)
            if scope(sa_result) < best_scope:
                best_rows = sa_result
                best_scope = scope(sa_result)

    # Iterated Local Search
    # Alternates between local polish, SA polish, and kicks
    while time_left() > 25:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        # 2. SA polish
        if time_left() > 15:
            sa_result = sa_polish(best_rows)
            if scope(sa_result) < best_scope:
                best_rows = sa_result
                best_scope = scope(sa_result)
        
        # 3. Kick
        if time_left() < 20:
            break
            
        kicked_rows = kick(best_rows)
        
        # 4. Polish kicked rows (both local and SA)
        polished_kicked = local_polish(kicked_rows)
        if time_left() > 12:
            polished_kicked = sa_polish(polished_kicked)
        
        # 5. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Final polish to ensure we squeeze out any last drops
    if time_left() > 8:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        if time_left() > 5:
            sa_final = sa_polish(best_rows)
            if scope(sa_final) < best_scope:
                best_rows = sa_final
                best_scope = scope(sa_final)

    # Final validity guarantee.
    if not is_valid(best_rows):
        # Fallback to a fresh greedy (always valid).
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
