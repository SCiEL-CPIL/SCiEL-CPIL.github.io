import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    best_rows = None
    best_scope = float('inf')
    
    # Helper: check if a row is valid given the set of used differences
    def _is_valid_row(row, used_diffs):
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in row_diffs:
                    return False
                if d in used_diffs:
                    return False
                row_diffs.add(d)
        return True
    
    # Helper: get all differences from a row
    def _get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Phase 1: Deterministic greedy baseline
    def _deterministic_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Phase 2: Randomized greedy with candidate selection
    def _random_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Find valid candidates
                candidates = []
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                        if len(candidates) >= 10:
                            break
                    x += 1
                    if x > 500:
                        break
                
                if not candidates:
                    return None
                
                # Pick randomly from candidates
                next_x = random.choice(candidates)
                row.append(next_x)
                for a in row[:-1]:
                    used.add(next_x - a)
                current_max = next_x
            
            rows.append(row)
        
        # Verify validity
        all_diffs = set()
        for row in rows:
            for d in _get_diffs(row):
                if d in all_diffs:
                    return None
                all_diffs.add(d)
        
        return rows
    
    # Phase 3: Local search - try to shrink individual rows
    def _try_shrink_row(row, used_diffs, max_attempts=100):
        """Try to reduce the maximum entry in a row while maintaining validity."""
        row = row[:]
        row_diffs = _get_diffs(row)
        
        # Try to reduce each position from right to left
        for pos in range(len(row) - 1, 0, -1):
            for attempt in range(max_attempts):
                # Try a smaller value
                target = row[pos] - 1
                if target <= row[pos - 1]:
                    break
                
                # Check if target is valid
                new_diffs = set()
                valid = True
                for i in range(pos):
                    d = target - row[i]
                    if d in new_diffs:
                        valid = False
                        break
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    # Also need to check that removing old diffs and adding new ones
                    # doesn't conflict with other rows' diffs (handled by used_diffs)
                    # Update the row
                    row[pos] = target
                    # Update used_diffs: remove old diffs involving position pos, add new ones
                    for i in range(pos):
                        old_d = row[pos] + 1 - row[i]  # This is tricky, let me redo
                        # Actually, we need to be more careful
                        pass
                    # Simpler: just rebuild
                    row_diffs_new = _get_diffs(row)
                    # Check no internal conflicts
                    if len(row_diffs_new) == len(set(row_diffs_new)):
                        # Check no conflicts with used_diffs (excluding this row's old diffs)
                        other_diffs = used_diffs - row_diffs
                        if not (row_diffs_new & other_diffs):
                            row_diffs = row_diffs_new
                            # Update used_diffs
                            used_diffs = (used_diffs - row_diffs) | row_diffs
                            # Continue trying to shrink this position
                            continue
                break
        
        return row
    
    # Phase 4: Full local search with row reordering
    def _local_search(rows, max_iters=200):
        """Try to improve the scope by shrinking rows."""
        rows = [r[:] for r in rows]
        all_diffs = set()
        for row in rows:
            all_diffs |= _get_diffs(row)
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            improved = False
            # Try to shrink each row
            for i in range(len(rows)):
                row = rows[i]
                other_diffs = all_diffs - _get_diffs(row)
                
                # Try to reduce the maximum entry
                max_val = max(row)
                for new_max in range(max_val - 1, max(row[:-1]), -1):
                    # Try to find a valid row with max entry = new_max
                    # This is complex; instead, try small perturbations
                    pass
                
                # Simpler: try to reduce individual positions
                for pos in range(len(row) - 1, 0, -1):
                    for delta in range(1, 5):
                        target = row[pos] - delta
                        if target <= row[pos - 1]:
                            break
                        
                        # Check validity
                        new_row = row[:]
                        new_row[pos] = target
                        new_diffs = _get_diffs(new_row)
                        
                        # Check no internal conflicts
                        if len(new_diffs) != len(set(new_diffs)):
                            continue
                        
                        # Check no conflicts with other rows
                        if new_diffs & other_diffs:
                            continue
                        
                        # Accept
                        rows[i] = new_row
                        all_diffs = (all_diffs - _get_diffs(row)) | new_diffs
                        improved = True
                        break
                    if improved:
                        break
            
            if not improved:
                break
        
        return rows
    
    # Run deterministic greedy first
    rows = _deterministic_greedy()
    if rows:
        scope = max(max(r) for r in rows)
        best_scope = scope
        best_rows = [r[:] for r in rows]
    
    # Run randomized greedy with multiple restarts
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        
        rows = _random_greedy(seed)
        if rows:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= 134:
                    break
    
    # Apply local search to best solution
    if best_rows:
        improved = _local_search(best_rows, max_iters=500)
        scope = max(max(r) for r in improved)
        if scope < best_scope:
            best_scope = scope
            best_rows = improved
    
    # More randomized restarts with local search
    for seed in range(50, 200):
        if time.time() > deadline - 5:
            break
        
        rows = _random_greedy(seed)
        if rows:
            improved = _local_search(rows, max_iters=200)
            scope = max(max(r) for r in improved)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in improved]
                if best_scope <= 134:
                    break
    
    # Final fallback
    if best_rows is None:
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
    
    return best_rows
