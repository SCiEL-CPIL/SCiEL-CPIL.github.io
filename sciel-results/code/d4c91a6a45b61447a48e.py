"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 145.0  # safety margin before 150s kill
    MAX_ENTRY_LIMIT = 200  # hard cap to keep search tractable

    # Helper: compute all pairwise differences for a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper: check if a set of diffs is disjoint from used
    def is_disjoint(diffs, used):
        for d in diffs:
            if d in used:
                return False
        return True

    # Helper: greedy construction of a single row given used diffs
    def greedy_row(used, max_entry=MAX_ENTRY_LIMIT):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_entry:
                return None
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        return row

    # Helper: try to rebuild a specific row with a different start/greedy strategy
    def rebuild_row(row_idx, rows, used, max_entry=MAX_ENTRY_LIMIT):
        """Try to rebuild row_idx to reduce the max entry."""
        # Remove current row's diffs from used
        old_diffs = row_diffs(rows[row_idx])
        used -= old_diffs
        
        old_max = rows[row_idx][-1]
        
        # Try greedy with various strategies
        best_row = None
        best_max = old_max
        
        # Strategy 1: pure greedy
        test_used = set(used)
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > best_max - 1:
                break
            new_diffs = {x - a for a in row}
            if not (new_diffs & test_used):
                row.append(x)
                test_used |= new_diffs
        if len(row) == k + 1 and row[-1] < best_max:
            best_row = row
            best_max = row[-1]
        
        # Strategy 2: greedy but skip some early candidates to find tighter pack
        for skip_pattern in range(min(5, k)):
            test_used = set(used)
            row = [0]
            x = 0
            skip_next = False
            while len(row) < k + 1:
                x += 1
                if x > best_max - 1:
                    break
                if skip_next:
                    skip_next = False
                    continue
                new_diffs = {x - a for a in row}
                if not (new_diffs & test_used):
                    # Occasionally skip to explore
                    if random.random() < 0.1 and len(row) < k:
                        skip_next = True
                        continue
                    row.append(x)
                    test_used |= new_diffs
            if len(row) == k + 1 and row[-1] < best_max:
                best_row = row
                best_max = row[-1]
        
        if best_row is not None:
            rows[row_idx] = best_row
            used |= row_diffs(best_row)
            return True
        else:
            # Restore
            used |= old_diffs
            return False

    # Initial greedy construction
    used = set()
    rows = []
    for i in range(n_rows):
        row = greedy_row(used)
        if row is None:
            # Fallback: allow larger entries
            row = greedy_row(used, max_entry=MAX_ENTRY_LIMIT * 2)
            if row is None:
                # Last resort: just return what we have
                break
        rows.append(row)
    
    if len(rows) < n_rows:
        # Pad with trivial rows if needed (shouldn't happen)
        while len(rows) < n_rows:
            rows.append(list(range(k + 1)))
    
    current_max = max(row[-1] for row in rows)
    best_rows = [list(r) for r in rows]
    best_max = current_max

    # Hill climbing: iteratively try to rebuild rows to reduce max
    iteration = 0
    while time.time() < DEADLINE:
        # Pick a row to rebuild: prioritize rows with large max
        row_order = sorted(range(n_rows), key=lambda i: -rows[i][-1])
        
        improved = False
        for row_idx in row_order:
            if time.time() > DEADLINE:
                break
            if rows[row_idx][-1] <= best_max:
                continue
            if rebuild_row(row_idx, rows, used):
                improved = True
                new_max = max(r[-1] for r in rows)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in rows]
        
        if not improved:
            # Try a more aggressive restart of one row
            row_idx = row_order[0]
            old_row = rows[row_idx]
            old_diffs = row_diffs(old_row)
            used -= old_diffs
            
            # Try multiple random restarts for this row
            best_candidate = None
            best_candidate_max = old_row[-1]
            
            for attempt in range(20):
                if time.time() > DEADLINE:
                    break
                test_used = set(used)
                row = [0]
                x = 0
                success = True
                while len(row) < k + 1:
                    x += 1
                    if x > best_candidate_max - 1:
                        success = False
                        break
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & test_used):
                        # With some probability, skip to explore
                        if random.random() < 0.15 and len(row) < k:
                            continue
                        row.append(x)
                        test_used |= new_diffs
                if success and len(row) == k + 1 and row[-1] < best_candidate_max:
                    best_candidate = row
                    best_candidate_max = row[-1]
            
            if best_candidate is not None:
                rows[row_idx] = best_candidate
                used |= row_diffs(best_candidate)
                new_max = max(r[-1] for r in rows)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in rows]
            else:
                used |= old_diffs
        
        iteration += 1
        if time.time() > DEADLINE:
            break

    return best_rows

# EVOLVE-BLOCK-END
