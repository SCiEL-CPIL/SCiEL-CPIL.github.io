import time
import random
import sys

def construct(n_rows, k):
    """
    Constructs n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct over all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (or 150s)
    
    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Helper to get all differences for a specific row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # 1. Randomized Greedy Construction
    def greedy_randomized(n_rows, k, window_size=30, rand_prob=0.15):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                start_val = row[-1] + 1
                valid_candidates = []
                
                # Limit search depth to avoid infinite loops in greedy phase
                # We look for the smallest valid x, but sometimes pick a larger one
                # Scanning a window of candidates
                found_any = False
                # Dynamic window: if we are stuck, scan further
                scan_limit = start_val + window_size * 10 
                for x in range(start_val, scan_limit):
                    new = {x - a for a in row}
                    if not (new & used):
                        valid_candidates.append(x)
                        if len(valid_candidates) >= window_size:
                            break
                        found_any = True
                
                if not valid_candidates:
                    # Fallback: scan indefinitely (should be rare with good window)
                    x = start_val
                    while True:
                        new = {x - a for a in row}
                        if not (new & used):
                            valid_candidates.append(x)
                            break
                        x += 1
                
                if valid_candidates:
                    if random.random() < rand_prob and len(valid_candidates) > 1:
                        x = random.choice(valid_candidates)
                    else:
                        x = valid_candidates[0]
                    
                    row.append(x)
                    used.update({x - a for a in row[:-1]})
                else:
                    break
            rows.append(row)
        return rows, used

    # 2. Targeted Local Optimization
    def local_optimize_targeted(rows, used, deadline):
        """
        Focuses on reducing the global maximum.
        If stuck, perturbs other elements to create space.
        """
        current_max = max(max(r) for r in rows)
        best_rows = [r[:] for r in rows]
        best_used = set(used)
        best_scope = current_max
        
        # Pre-compute diffs for each row to speed up conflict checks if needed, 
        # but we rely on the global 'used' set for validity.
        
        def is_valid_move(row_idx, pos, new_val, used_set, rows_list):
            """
            Checks if setting rows_list[row_idx][pos] to new_val is valid
            given the current used_set (which includes diffs for the OLD value at pos).
            We must remove old diffs, add new diffs, and check for conflicts.
            """
            row = rows_list[row_idx]
            old_val = row[pos]
            
            # 1. Identify old diffs to remove
            old_diffs = set()
            for i in range(row_len):
                if i != pos:
                    if old_val > row[i]:
                        old_diffs.add(old_val - row[i])
                    else:
                        old_diffs.add(row[i] - old_val)
            
            # 2. Identify new diffs to add
            new_diffs = set()
            for i in range(row_len):
                if i != pos:
                    if new_val > row[i]:
                        new_diffs.add(new_val - row[i])
                    else:
                        new_diffs.add(row[i] - new_val)
            
            # 3. Check for internal duplicates in new_diffs
            # Since row is strictly increasing, new_val must be between row[pos-1] and row[pos+1]
            if pos > 0 and new_val <= row[pos-1]:
                return False
            if pos < row_len - 1 and new_val >= row[pos+1]:
                return False
            
            # Check if any new_diff is in (used_set - old_diffs)
            # If it is, it means it conflicts with a diff from another row or a non-removed diff in this row.
            # Note: used_set currently contains old_diffs.
            # So we check: new_diffs & (used_set - old_diffs)
            
            # Optimization: We can just check if any d in new_diffs is in used_set 
            # AND d is not in old_diffs.
            
            # However, we must also ensure that new_diffs doesn't contain duplicates itself.
            # Since row is increasing, diffs to left are new_val - row[i] (positive) and diffs to right are row[j] - new_val (positive).
            # Can a left diff equal a right diff? 
            # new_val - row[i] == row[j] - new_val => 2*new_val == row[i] + row[j]. Possible.
            # So we check len(set(new_diffs)) == len(new_diffs).
            if len(new_diffs) != len(new_diffs):
                return False
            
            # Check conflicts with global used set
            # We need to remove old_diffs from consideration
            # If d in new_diffs and d in used_set and d not in old_diffs -> Conflict
            
            # Efficient check:
            for d in new_diffs:
                if d in used_set and d not in old_diffs:
                    return False
            
            return True

        def apply_move(rows_list, used_set, row_idx, pos, new_val):
            """
            Applies a move, updating rows_list and used_set.
            Assumes move is valid.
            """
            row = rows_list[row_idx]
            old_val = row[pos]
            
            old_diffs = set()
            for i in range(row_len):
                if i != pos:
                    if old_val > row[i]:
                        old_diffs.add(old_val - row[i])
                    else:
                        old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(row_len):
                if i != pos:
                    if new_val > row[i]:
                        new_diffs.add(new_val - row[i])
                    else:
                        new_diffs.add(row[i] - new_val)
                        
            used_set.difference_update(old_diffs)
            used_set.update(new_diffs)
            row[pos] = new_val

        while time.time() < deadline:
            # 1. Find global maximum
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            
            for r_idx, r in enumerate(rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            
            if max_c_idx == 0:
                # Cannot change 0
                break
                
            current_val = max_val
            prev_val = rows[max_r_idx][max_c_idx - 1]
            next_val = rows[max_r_idx][max_c_idx + 1] if max_c_idx < row_len - 1 else current_val + 1000
            
            # 2. Try to reduce the maximum
            found_reduction = False
            # Scan downwards from current_val - 1 to prev_val + 1
            # We want the smallest valid value? No, we just want ANY valid value < current_val
            # to reduce the scope. If we find one, we apply it.
            # To be more aggressive, we could try to find the smallest valid value to maximize future reductions,
            # but that's expensive. Let's try a few random steps down or linear down.
            
            # Linear down is safer for correctness.
            for new_val in range(current_val - 1, prev_val, -1):
                if is_valid_move(max_r_idx, max_c_idx, new_val, used, rows):
                    apply_move(rows, used, max_r_idx, max_c_idx, new_val)
                    found_reduction = True
                    # Update best
                    new_scope = max(max(r) for r in rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in rows]
                        best_used = set(used)
                    break
            
            if not found_reduction:
                # 3. Perturbation: Try to move a different element to create space
                # Pick a random row and column (not 0, not the current max if possible)
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, row_len - 1)
                
                # If we picked the max, pick another
                if r_idx == max_r_idx and c_idx == max_c_idx:
                    c_idx = random.randint(1, row_len - 1)
                    if r_idx == max_r_idx and c_idx == max_c_idx:
                        r_idx = (r_idx + 1) % n_rows
                        c_idx = random.randint(1, row_len - 1)
                
                curr_val = rows[r_idx][c_idx]
                prev_v = rows[r_idx][c_idx - 1]
                next_v = rows[r_idx][c_idx + 1] if c_idx < row_len - 1 else curr_val + 50
                
                # Try to move this element to a random valid position nearby
                # Range: prev_v + 1 to min(curr_val + 10, next_v - 1)
                # We want to "shake" the system.
                lower_bound = prev_v + 1
                upper_bound = min(curr_val + 10, next_v - 1)
                
                if lower_bound <= upper_bound:
                    # Try a few random values in this range
                    attempts = 0
                    while attempts < 5 and time.time() < deadline:
                        new_val = random.randint(lower_bound, upper_bound)
                        if new_val != curr_val:
                            if is_valid_move(r_idx, c_idx, new_val, used, rows):
                                apply_move(rows, used, r_idx, c_idx, new_val)
                                break
                        attempts += 1
                else:
                    # If no space nearby, try a larger jump? 
                    # For now, just skip perturbation if stuck locally
                    pass

        return best_rows, best_scope

    best_rows = None
    best_score = float('inf')
    best_used = None
    
    # Run multiple randomized greedy starts
    attempts = 0
    # We will spend ~20% of time on init and 80% on optimization
    # But since optimization is the main driver, we do a few quick inits
    # and then optimize the best one found so far.
    
    # Strategy: 
    # 1. Generate 1 initial solution.
    # 2. Optimize it for a chunk of time.
    # 3. If time remains, generate another, optimize it, keep best.
    
    # However, local_optimize_targeted runs until deadline. 
    # We need to pass a sub-deadline to allow multiple restarts.
    
    # Let's do 3 restarts.
    num_restarts = 3
    time_per_restart = (deadline - start_time) / num_restarts
    
    for i in range(num_restarts):
        if time.time() >= deadline - 5:
            break
            
        window = random.randint(20, 40)
        prob = random.uniform(0.1, 0.2)
        
        rows, used = greedy_randomized(n_rows, k, window, prob)
        
        score = max(max(r) for r in rows)
        
        if score < best_score:
            best_score = score
            best_rows = [r[:] for r in rows]
            best_used = set(used)
            
        # Optimize this specific solution
        # Give it the remaining time for this restart
        restart_deadline = start_time + (i + 1) * time_per_restart
        # But don't exceed global deadline
        restart_deadline = min(restart_deadline, deadline - 2)
        
        opt_rows, opt_score = local_optimize_targeted(rows, used, restart_deadline)
        
        if opt_score < best_score:
            best_score = opt_score
            best_rows = opt_rows
            # We don't strictly need best_used if we return best_rows, 
            # but keeping it helps if we were to continue, though we don't.
            
    if best_rows is None:
        # Fallback to simple greedy if all failed (should not happen)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not (new & used):
                        break
                    x += 1
                row.append(x)
                used |= new
            rows.append(row)
        best_rows = rows

    return best_rows
