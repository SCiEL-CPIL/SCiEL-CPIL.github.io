import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    def get_row_diffs(row):
        diffs = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                diffs |= (1 << d)
        return diffs

    def is_valid_row(row, global_used_except_self):
        # Check internal distinctness and external distinctness
        row_diffs = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if (row_diffs >> d) & 1:
                    return False, 0 # Internal conflict
                row_diffs |= (1 << d)
        if row_diffs & global_used_except_self:
            return False, 0 # External conflict
        return True, row_diffs

    def greedy_row(max_val, global_used, existing_vals=None):
        """
        Greedily construct a row of length k+1 starting with 0, 
        with all values <= max_val, avoiding conflicts with global_used.
        existing_vals: if provided, these are fixed prefix values.
        """
        if existing_vals is None:
            row = [0]
        else:
            row = list(existing_vals)
            
        # Calculate diffs from existing row
        current_diffs = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                current_diffs |= (1 << (row[j] - row[i]))
                
        # Check if existing row is already valid internally
        # (Assuming it is, as we build incrementally)
        
        while len(row) < k + 1:
            last = row[-1]
            found = False
            # Try smallest possible next value
            for v in range(last + 1, max_val + 1):
                # Check conflicts with current row
                conflict = False
                new_diffs = 0
                for u in row:
                    d = v - u
                    if (current_diffs >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                
                if conflict:
                    continue
                
                # Check conflicts with global
                if new_diffs & global_used:
                    continue
                
                # Accept
                row.append(v)
                current_diffs |= new_diffs
                found = True
                break
            
            if not found:
                return None
        return row

    def random_row(max_val, global_used, attempts=50):
        """
        Try to find a valid row with max <= max_val using randomized greedy.
        """
        for _ in range(attempts):
            row = [0]
            current_diffs = 0
            success = True
            for _ in range(k):
                last = row[-1]
                candidates = list(range(last + 1, max_val + 1))
                random.shuffle(candidates)
                found = False
                for v in candidates:
                    conflict = False
                    new_diffs = 0
                    for u in row:
                        d = v - u
                        if (current_diffs >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    if conflict:
                        continue
                    if new_diffs & global_used:
                        continue
                    row.append(v)
                    current_diffs |= new_diffs
                    found = True
                    break
                if not found:
                    success = False
                    break
            if success:
                # Final check
                valid, _ = is_valid_row(row, global_used)
                if valid:
                    return row
        return None

    # 1. Initial Solution: Greedy with large scope
    # Estimate a safe starting scope. Lower bound is 112. Let's start at 200.
    init_scope = 250
    rows = []
    global_used = 0
    success = False
    for _ in range(50): # Try a few times with randomization if greedy fails
        rows = []
        global_used = 0
        valid = True
        for i in range(n_rows):
            row = greedy_row(init_scope, global_used)
            if row is None:
                valid = False
                break
            rows.append(row)
            global_used |= get_row_diffs(row)
        if valid:
            success = True
            break
    
    if not success:
        # Fallback: Very large scope
        init_scope = 500
        rows = []
        global_used = 0
        for i in range(n_rows):
            row = greedy_row(init_scope, global_used)
            if row is None:
                # This should not happen with scope 500
                row = [0] + list(range(1, k+1)) # Invalid but placeholder
            rows.append(row)
            global_used |= get_row_diffs(row)
            
    best_rows = [list(r) for r in rows]
    best_max = get_max_entry(best_rows)
    current_rows = [list(r) for r in rows]
    current_max = best_max
    current_global_used = global_used

    # 2. Optimization Loop: Shrink Scope
    # We try to reduce the maximum entry of the current solution.
    
    def try_shrink_row(row_idx, target_max, global_used_except_self):
        """
        Try to replace current_rows[row_idx] with a new row that has max <= target_max.
        Returns the new row if found, else None.
        """
        # Use randomized greedy to find a row within target_max
        # We can bias towards smaller values
        for _ in range(20): # 20 attempts
            row = random_row(target_max, global_used_except_self, attempts=1)
            if row is not None:
                return row
        return None

    # Main loop
    step = 0
    no_improve_count = 0
    
    while time.time() < deadline - 5.0:
        step += 1
        
        # Calculate global used except for the row we are modifying
        # We pick a random row to modify
        row_idx = random.randint(0, n_rows - 1)
        
        # Calculate global used for other rows
        global_used_except = 0
        for i in range(n_rows):
            if i != row_idx:
                global_used_except |= get_row_diffs(current_rows[i])
                
        curr_row_max = current_rows[row_idx][-1]
        
        # Target: reduce curr_row_max
        # Try to shrink by 1, 2, 5, 10...
        # If we can shrink, we update.
        improved = False
        
        # Try to find a row with max < curr_row_max
        # We can try specific targets: curr_row_max - 1, curr_row_max - 5, etc.
        # Or just try to find ANY valid row with max < curr_row_max
        # Since random_row is stochastic, we can just call it with target = curr_row_max - 1
        # If it fails, try curr_row_max - 2, etc.
        
        # To be efficient, let's try a few random targets below curr_row_max
        # Or just try to shrink by 1 repeatedly.
        
        # Strategy: Try to shrink by 1. If stuck for many steps, try a bigger jump or restart.
        target = curr_row_max - 1
        
        # If target is too small (below lower bound for this row given others), it will fail.
        # Lower bound for a single row is k(k+1)/2 = 28.
        # But global constraints matter.
        
        new_row = try_shrink_row(row_idx, target, global_used_except)
        
        if new_row is not None:
            # Validate new row against global
            valid, new_diffs = is_valid_row(new_row, global_used_except)
            if valid:
                # Update state
                current_rows[row_idx] = new_row
                # Recalculate global used
                current_global_used = 0
                for i in range(n_rows):
                    current_global_used |= get_row_diffs(current_rows[i])
                
                new_global_max = get_max_entry(current_rows)
                if new_global_max < best_max:
                    best_max = new_global_max
                    best_rows = [list(r) for r in current_rows]
                    no_improve_count = 0
                else:
                    # Scope didn't improve globally, but this row is smaller.
                    # This might help future shrinks of other rows.
                    # Keep the change? Yes, it's a valid state.
                    pass
                improved = True
        
        if not improved:
            no_improve_count += 1
            
            # If stuck for a while, try a "perturbation":
            # Regenerate a random row with a larger scope to escape local minima,
            # then try to shrink it.
            if no_improve_count > 50 and step % 10 == 0:
                # Pick a row to perturb
                pert_idx = random.randint(0, n_rows - 1)
                # Use a scope slightly larger than current max to ensure we can find a valid row
                pert_scope = current_max + 10 
                global_used_except_pert = 0
                for i in range(n_rows):
                    if i != pert_idx:
                        global_used_except_pert |= get_row_diffs(current_rows[i])
                        
                new_pert_row = random_row(pert_scope, global_used_except_pert, attempts=10)
                if new_pert_row is not None:
                    valid, _ = is_valid_row(new_pert_row, global_used_except_pert)
                    if valid:
                        current_rows[pert_idx] = new_pert_row
                        # Recalculate global used
                        current_global_used = 0
                        for i in range(n_rows):
                            current_global_used |= get_row_diffs(current_rows[i])
                        # Note: current_max might increase, but we keep best_rows safe.
                        # We update current_max to reflect the new state
                        current_max = get_max_entry(current_rows)
                        no_improve_count = 0 # Reset stagnation counter after perturbation

    # Final Validation
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    if not validate(best_rows):
        # Fallback to a known valid structure if something went wrong
        # This should not happen if is_valid_row is correct
        # But just in case, return a valid large solution
        # Greedy with large scope
        rows_fallback = []
        gu = 0
        for i in range(n_rows):
            r = greedy_row(1000, gu)
            if r is None:
                r = [0] + list(range(1, k+1))
            rows_fallback.append(r)
            gu |= get_row_diffs(r)
        best_rows = rows_fallback

    return best_rows
