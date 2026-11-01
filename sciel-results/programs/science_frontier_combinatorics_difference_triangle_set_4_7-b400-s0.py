import time
import random

def construct(n_rows, k):
    # n_rows=4, k=7 (8 numbers)
    # We want to minimize M (max entry)
    
    start_time = time.time()
    # Safety margin for the 120s limit. Let's aim for 115s of work.
    deadline = start_time + 115.0

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def row_diffs_set(row):
        """All positive differences in a row (as a set)."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
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
    def try_reduce_row(rows, row_idx, used_other, max_attempts=150):
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
            # Jitter increases with attempts to explore more
            jitter = 0.15 if attempt > 15 else 0.0
            while len(row) < k + 1:
                found = False
                # Collect all valid candidates in the window
                candidates = []
                for trial_x in range(x + 1, x + 80):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        # Score: prefer smaller values, but also consider future flexibility
                        # A value that creates differences close to existing used_other diffs
                        # might be more restrictive in the future
                        score = trial_x
                        # Add a small penalty if this value would create diffs that are
                        # close to boundaries of used_other (heuristic for flexibility)
                        candidates.append((score, trial_x, new))
                
                if not candidates:
                    success = False
                    break
                
                # Sort by score (which is just the value here, so smallest first)
                candidates.sort(key=lambda c: c[0])
                
                # Apply jitter: sometimes skip the best candidate to explore
                chosen_idx = 0
                if jitter > 0 and len(candidates) > 1:
                    # With probability jitter, pick a random candidate from top 5
                    if random.random() < jitter:
                        top_n = min(5, len(candidates))
                        chosen_idx = random.randint(0, top_n - 1)
                    else:
                        chosen_idx = 0
                
                _, chosen_x, chosen_new = candidates[chosen_idx]
                row.append(chosen_x)
                current_row_diffs |= chosen_new
                found = True
                x = row[-1]
            if success and max(row) < best_max:
                # Verify the whole set is valid with this new row
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
        max_passes = 6
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            # Sort indices by descending max entry to target the largest rows first
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=80)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def kick(rows):
        """Perturb the solution to escape local minima."""
        rows = [list(r) for r in rows]
        # Strategy: Rebuild 1 or 2 random rows with high randomness
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            # Calculate used diffs from other rows
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            # Rebuild the row with high jitter
            success = False
            # Try a few times to find a valid row, even if larger
            for attempt in range(25):
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                while len(row) < k + 1:
                    found = False
                    # Collect candidates and use conflict-aware selection
                    candidates = []
                    for trial_x in range(x + 1, x + 120):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            # For kicks, we want more diversity
                            # Score: bias toward smaller but with more randomness
                            score = trial_x + random.random() * 20
                            candidates.append((score, trial_x, new))
                    
                    if not candidates:
                        valid_row = False
                        break
                    
                    # Sort by score
                    candidates.sort(key=lambda c: c[0])
                    
                    # High jitter: pick from top 10 with randomness
                    if len(candidates) > 1 and random.random() < 0.45:
                        top_n = min(10, len(candidates))
                        chosen_idx = random.randint(0, top_n - 1)
                    else:
                        chosen_idx = 0
                    
                    _, chosen_x, chosen_new = candidates[chosen_idx]
                    row.append(chosen_x)
                    current_row_diffs |= chosen_new
                    found = True
                    x = row[-1]
                
                if valid_row:
                    rows[idx] = row
                    success = True
                    break
            
            if not success:
                # If we failed to rebuild, keep the original row
                # This might make the solution invalid temporarily, but local_polish will fix it
                # Or we revert to a greedy row
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

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search
    # We use a loop that alternates between Local Search and Kicks
    # This is structurally different from a single long SA run.
    while time_left() > 20:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        # 2. Kick
        if time_left() < 15:
            break
            
        kicked_rows = kick(best_rows)
        
        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)
        
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)
        elif time_left() > 30:
            # Additional refinement: try multiple kicks and pick the best
            best_kicked = polished_kicked
            for extra_kick in range(3):
                if time_left() < 15:
                    break
                kicked2 = kick(best_rows)
                polished2 = local_polish(kicked2)
                if scope(polished2) < scope(best_kicked):
                    best_kicked = polished2
            if scope(best_kicked) < best_scope:
                best_rows = best_kicked
                best_scope = scope(best_kicked)

    # Final high-precision polish pass with tighter parameters
    if time_left() > 8:
        final_polish = local_polish(best_rows)
        if scope(final_polish) < best_scope:
            best_rows = final_polish
            best_scope = scope(final_polish)

    return best_rows
