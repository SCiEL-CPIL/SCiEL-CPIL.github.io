import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def greedy_construct(seed_val):
        """Construct rows using greedy with a compact strategy."""
        random.seed(seed_val)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                    x += 1
                
                if not found:
                    return None, float('inf')
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return rows, scope

    # 1. Get a valid starting point
    best_initial_rows = None
    best_initial_scope = float('inf')
    
    for seed in range(3):
        if time.time() > deadline - 40:
            break
        rows, scope = greedy_construct(seed)
        if rows is not None and scope < best_initial_scope:
            best_initial_scope = scope
            best_initial_rows = [list(r) for r in rows]
            
    if best_initial_rows is None:
        rows, scope = greedy_construct(0)
        if rows is None:
            used = set()
            best_initial_rows = []
            for _ in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    x = row[-1] + 1
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used:
                                valid = False
                                break
                        if valid:
                            break
                    row.append(x)
                    used.update({x - a for a in row if a != x})
                best_initial_rows.append(row)
            best_initial_scope = max(max(r) for r in best_initial_rows)
        else:
            best_initial_rows = [list(r) for r in rows]
            best_initial_scope = scope

    current_rows = [list(r) for r in best_initial_rows]
    current_used = get_used_diffs(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = best_initial_scope

    # 2. Targeted Backtracking: Try to find a valid configuration with scope < best_scope
    # Use bitset for fast constraint checking
    
    def try_backtrack_for_scope(target_scope, seed_val):
        """Try to construct all rows with max element <= target_scope."""
        random.seed(seed_val)
        
        # Use bitset for used differences
        # Differences can be up to target_scope
        used_bits = 0  # Bit i set means difference i is used
        
        def is_diff_used(d):
            if d <= 0 or d > target_scope:
                return True
            return (used_bits >> d) & 1
        
        def mark_diff(d):
            nonlocal used_bits
            if 0 < d <= target_scope:
                used_bits |= (1 << d)
        
        def unmark_diff(d):
            nonlocal used_bits
            if 0 < d <= target_scope:
                used_bits &= ~(1 << d)
        
        rows = []
        
        def backtrack_row(row_idx, current_row):
            if len(current_row) == k + 1:
                rows.append(list(current_row))
                return True
            
            if time.time() > deadline - 5:
                return False
            
            last = current_row[-1] if current_row else 0
            start_val = last + 1
            end_val = min(target_scope, last + (k + 1 - len(current_row)))
            
            # Try values in randomized order for diversity
            candidates = list(range(start_val, end_val + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                # Check if x is valid
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if is_diff_used(d):
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Try this value
                current_row.append(x)
                for d in new_diffs:
                    mark_diff(d)
                
                if backtrack_row(row_idx, current_row):
                    return True
                
                # Backtrack
                for d in new_diffs:
                    unmark_diff(d)
                current_row.pop()
            
            return False
        
        # Try to construct all rows
        for r_idx in range(n_rows):
            if not backtrack_row(r_idx, [0]):
                return None
        
        return rows
    
    # 3. Main loop: try to reduce scope using backtracking
    search_scope = best_scope
    
    while time.time() < deadline - 10:
        # Try backtracking with current search_scope
        found = False
        for seed in range(2):
            if time.time() > deadline - 10:
                break
            rows = try_backtrack_for_scope(search_scope - 1, seed)
            if rows is not None:
                # Verify
                used = get_used_diffs(rows)
                if len(used) == total_diffs:
                    new_scope = max(max(r) for r in rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                        search_scope = new_scope
                        found = True
                        break
        
        if not found:
            # Try to improve using local search on current best
            # Simple hill climbing: try to reduce individual elements
            improved = False
            for r_idx in range(n_rows):
                for p_idx in range(len(best_rows[r_idx]) - 1, 0, -1):
                    old_val = best_rows[r_idx][p_idx]
                    if old_val <= 1:
                        continue
                    
                    # Try to move to smaller value
                    for cand in range(old_val - 1, max(old_val - 5, 0), -1):
                        # Check if cand is valid
                        row = best_rows[r_idx]
                        valid = True
                        new_diffs = []
                        for a in row:
                            if a == old_val:
                                continue
                            d = abs(a - cand)
                            # Check if d is used by other elements in this row or other rows
                            # Simplified: just check against current_used
                            if d in current_used and d != abs(a - old_val):
                                valid = False
                                break
                            new_diffs.append(d)
                        
                        if valid:
                            # Check no conflict with other rows
                            # This is simplified; in practice need full check
                            break
            
            # If no improvement after trying, break
            if not improved and search_scope <= lb:
                break
            elif not improved:
                # Try a few more random local moves
                for _ in range(10):
                    if time.time() > deadline - 10:
                        break
                    r_idx = random.randint(0, n_rows - 1)
                    p_idx = random.randint(1, len(best_rows[r_idx]) - 1)
                    old_val = best_rows[r_idx][p_idx]
                    if old_val <= 1:
                        continue
                    
                    for cand in range(1, old_val):
                        row = best_rows[r_idx]
                        if cand in row:
                            continue
                        
                        valid = True
                        for a in row:
                            if a == old_val:
                                continue
                            d = abs(a - cand)
                            # Remove old diffs involving old_val
                            old_d = abs(a - old_val)
                            if d in current_used and d != old_d:
                                valid = False
                                break
                        
                        if valid:
                            # Accept move (simplified)
                            break
            
            search_scope = max(search_scope - 1, lb)
        
        if search_scope <= lb:
            break

    return best_rows
