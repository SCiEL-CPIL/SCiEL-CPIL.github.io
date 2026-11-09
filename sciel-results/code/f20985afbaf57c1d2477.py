"""Combined approach: bitset-accelerated greedy construction with randomized restarts
and local hill-climbing repair to minimize the scope (largest entry) for
difference triangle set (4,7)."""
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145.0  # Safety margin against 150s kill
    
    k_plus_1 = k + 1  # Number of elements per row
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row
    
    def get_scope(rows):
        if not rows:
            return 0
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Checks if the rows form a valid difference triangle set."""
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
    
    def row_diffs(row):
        """Compute all positive differences in a row as a set."""
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    def greedy_build_row(used_set, k, max_val, rng=None):
        """Build a single row greedily, trying smallest values first."""
        row = [0]
        while len(row) < k_plus_1:
            found = False
            for x in range(row[-1] + 1, max_val + 1):
                new_diffs = set()
                conflict = False
                for a in row:
                    d = x - a
                    if d > max_val or d in used_set:
                        conflict = True
                        break
                    new_diffs.add(d)
                if not conflict:
                    row.append(x)
                    used_set.update(new_diffs)
                    found = True
                    break
            if not found:
                return None
        return row
    
    def greedy_construct(n_rows, k, max_val, rng):
        """Build all rows greedily within max_val scope using randomized row order."""
        used = set()
        rows = []
        order = list(range(n_rows))
        rng.shuffle(order)
        for _ in order:
            row = greedy_build_row(used, k, max_val, rng)
            if row is None:
                return None
            rows.append(row)
        return rows
    
    # Phase 1: Get a baseline solution with standard greedy
    rng = random.Random(42)
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k_plus_1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(best_rows)
    
    # Phase 2: Try to improve scope using randomized greedy with tighter bounds
    # Try scopes from best_scope-1 down to a reasonable lower bound
    # Use the counting bound as a rough lower bound: total_diffs_needed <= scope*(scope+1)/2
    # So scope >= ceil((-1 + sqrt(1 + 8*total_diffs_needed)) / 2)
    import math
    lower_bound = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2))
    
    for scope_try in range(best_scope - 1, max(lower_bound, 100), -1):
        if time.time() > time_limit:
            break
        # Try multiple random orderings for this scope
        found_at_scope = False
        for attempt in range(30):
            if time.time() > time_limit:
                break
            result = greedy_construct(n_rows, k, scope_try, rng)
            if result is not None:
                if is_valid(result):
                    best_rows = [r[:] for r in result]
                    best_scope = scope_try
                    found_at_scope = True
                    break
        if not found_at_scope:
            # If we can't fit in this scope after several tries, 
            # it's likely we can't fit in smaller scopes either
            # But let's try a couple more scopes down to be safe
            if scope_try - best_scope < 5:
                break
    
    # Phase 3: Local search / hill climbing to reduce scope
    if time.time() < time_limit:
        rows_local = [list(r) for r in best_rows]
        local_best_scope = get_scope(rows_local)
        
        # Pre-compute differences for each row
        row_diffs_list = [row_diffs(r) for r in rows_local]
        
        def all_other_diffs(exclude_idx):
            """Get all differences except from row exclude_idx."""
            all_diffs = set()
            for i in range(n_rows):
                if i != exclude_idx:
                    all_diffs.update(row_diffs_list[i])
            return all_diffs
        
        improved = True
        iterations = 0
        max_iterations = 2000
        while improved and time.time() < time_limit and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for r_idx in range(n_rows):
                if time.time() > time_limit:
                    break
                row = rows_local[r_idx]
                # Try to move each mark (from last to first) to a smaller position
                for pos in range(k, 0, -1):
                    if time.time() > time_limit:
                        break
                    old_val = row[pos]
                    lower = row[pos - 1] + 1
                    # Try to move to a smaller value
                    for new_val in range(lower, old_val):
                        # Tentatively set the new value
                        tent_row = list(row)
                        tent_row[pos] = new_val
                        
                        # Check strictly increasing
                        valid = True
                        for i in range(len(tent_row) - 1):
                            if tent_row[i + 1] <= tent_row[i]:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Compute differences of tentative row
                        tent_diffs = row_diffs(tent_row)
                        
                        # Check no conflict with other rows
                        other_diffs = all_other_diffs(r_idx)
                        if tent_diffs & other_diffs:
                            continue
                        
                        # Accept the move
                        rows_local[r_idx] = tent_row
                        row_diffs_list[r_idx] = tent_diffs
                        new_scope = get_scope(rows_local)
                        if new_scope < local_best_scope:
                            local_best_scope = new_scope
                            improved = True
                        break  # Move to next position
            
            if local_best_scope < best_scope:
                best_rows = [list(r) for r in rows_local]
                best_scope = local_best_scope
        
        # Phase 4: Try row-replacement moves
        if time.time() < time_limit:
            for r_idx in range(n_rows):
                if time.time() > time_limit:
                    break
                # Try to replace row r_idx with a fresh greedy row using remaining diffs
                for attempt in range(10):
                    if time.time() > time_limit:
                        break
                    other_diffs = set()
                    for i in range(n_rows):
                        if i != r_idx:
                            other_diffs.update(row_diffs_list[i])
                    
                    # Build a new row avoiding other_diffs
                    new_row = [0]
                    while len(new_row) < k_plus_1:
                        found = False
                        for x in range(new_row[-1] + 1, best_scope + 1):
                            new_diffs = set()
                            conflict = False
                            for a in new_row:
                                d = x - a
                                if d > best_scope or d in other_diffs:
                                    conflict = True
                                    break
                                new_diffs.add(d)
                            if not conflict:
                                new_row.append(x)
                                other_diffs.update(new_diffs)
                                found = True
                                break
                        if not found:
                            break
                    
                    if len(new_row) == k_plus_1:
                        new_scope = get_scope([rows_local[i] if i != r_idx else new_row for i in range(n_rows)])
                        if new_scope < local_best_scope:
                            rows_local[r_idx] = new_row
                            row_diffs_list[r_idx] = row_diffs(new_row)
                            local_best_scope = new_scope
                            if local_best_scope < best_scope:
                                best_rows = [list(r) for r in rows_local]
                                best_scope = local_best_scope
                            break
    
    # Final validation
    if not is_valid(best_rows):
        # Fallback to original greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k_plus_1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    return best_rows
