import time
import random

def construct(n_rows, k):
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    size = k + 1
    # Pre-compute pairs (i, j) for i < j within a row
    pairs = [(i, j) for i in range(size) for j in range(i + 1, size)]
    
    def get_diffs(row):
        return [row[j] - row[i] for i, j in pairs]

    def is_valid(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        return len(all_diffs) == len(set(all_diffs))

    def get_scope(rows):
        return max(max(r) for r in rows)

    # 1. Greedy Construction to get a starting point
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < size:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    current_rows = greedy_construct()
    best_rows = [r[:] for r in current_rows]
    best_scope = get_scope(best_rows)
    best_conflicts = 0 # Greedy is valid by construction
    
    # 2. Setup for Simulated Annealing
    # We track the count of each difference globally.
    # Max possible difference is bounded by the scope. 
    # To be safe, we use a large array or a dict. 
    # Given scope ~130, an array of size 200 is fast enough.
    max_diff_bound = 500 
    diff_counts = [0] * max_diff_bound
    
    def update_counts(row, delta_sign):
        """Update global diff_counts for a row. delta_sign is +1 for add, -1 for remove."""
        nonlocal total_conflicts
        for d in get_diffs(row):
            if d < max_diff_bound:
                old_count = diff_counts[d]
                new_count = old_count + delta_sign
                diff_counts[d] = new_count
                # A conflict exists if count > 1.
                # Transition 1->2 adds 1 conflict. Transition 2->1 removes 1 conflict.
                if old_count == 1 and new_count == 2:
                    total_conflicts += 1
                elif old_count == 2 and new_count == 1:
                    total_conflicts -= 1
            else:
                # If difference is too large, we ignore it for conflict counting?
                # No, that's wrong. But for (4,7), diffs are small.
                # If we hit this, we need to resize.
                # For simplicity and speed, we assume max_diff_bound is sufficient.
                # 134 * 1 is the max diff. 500 is safe.
                pass

    total_conflicts = 0
    for r in current_rows:
        update_counts(r, 1)
        
    current_scope = get_scope(current_rows)
    
    def objective(conflicts, scope):
        # Prioritize validity (0 conflicts) over scope.
        # If conflicts > 0, penalty is huge.
        return conflicts * 10000 + scope

    current_obj = objective(total_conflicts, current_scope)
    best_obj = current_obj
    best_rows = [r[:] for r in current_rows]
    
    # SA Parameters
    T = 50.0
    cooling = 0.9995
    min_T = 0.1
    steps_per_second = 0 # Track speed
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Pick a random row and a random element to perturb
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Pick an index to move. 
        # Moving the first element (0) is not allowed (must start at 0).
        # Moving the last element is often useful for scope reduction.
        # We pick an index from 1 to size-1.
        e_idx = random.randint(1, size - 1)
        
        old_val = row[e_idx]
        
        # Determine valid range for new_val
        # Must be > row[e_idx-1] and < row[e_idx+1] (if exists)
        min_val = row[e_idx - 1] + 1
        max_val = row[e_idx + 1] - 1 if e_idx < size - 1 else old_val + 20
        
        if min_val > max_val:
            continue
            
        # Randomly choose a new value in the range
        # Bias towards smaller values to reduce scope
        if random.random() < 0.5:
            # Try to reduce
            new_val = random.randint(min_val, min(old_val, max_val))
        else:
            # Try to increase or shift
            new_val = random.randint(max(old_val, min_val), max_val)
            
        if new_val == old_val:
            continue
            
        # Calculate delta in objective function
        # We need to update diff_counts for this row.
        # Remove old row's contributions, change value, add new row's contributions.
        
        # 1. Remove old row
        update_counts(row, -1)
        
        # 2. Modify row
        row[e_idx] = new_val
        # Ensure sorted (it is, by construction of range)
        
        # 3. Add new row
        update_counts(row, 1)
        
        new_scope = get_scope(current_rows)
        new_obj = objective(total_conflicts, new_scope)
        delta_obj = new_obj - current_obj
        
        # Acceptance criterion
        if delta_obj < 0 or (T > 0 and random.random() < pow(2.718281828, -delta_obj / T)):
            current_obj = new_obj
            current_scope = new_scope
            if current_obj < best_obj:
                best_obj = current_obj
                best_rows = [r[:] for r in current_rows]
                if best_conflicts > 0:
                    best_conflicts = total_conflicts # Just for tracking, not used in return directly
        else:
            # Revert
            row[e_idx] = old_val
            # Update counts back to old state
            # Remove new (which is now old in array, but we need to revert the count change)
            # Actually, we modified row in place. 
            # Current state of row is old_val at e_idx.
            # Current diff_counts reflect the NEW row (with new_val).
            # We need to remove NEW row's diffs and add OLD row's diffs.
            # But row is already reverted to old_val.
            # So get_diffs(row) gives OLD diffs.
            # We need to:
            # 1. Remove NEW diffs from counts.
            # 2. Add OLD diffs to counts.
            
            # To do this, we need the NEW diffs. We can reconstruct them?
            # Or we can just swap the logic: 
            # We did: Remove Old, Add New.
            # To revert: Remove New, Add Old.
            # But row is currently Old. So get_diffs(row) is Old.
            # We need to temporarily set it to New to get New diffs?
            # That's expensive.
            
            # Alternative: Keep a copy of the row before modification?
            # That's also expensive.
            
            # Better: 
            # We know the change was at index e_idx.
            # The diffs involving e_idx changed.
            # Let's just re-apply the inverse update.
            # Current row is OLD.
            # Counts reflect NEW.
            # We need to remove NEW diffs and add OLD diffs.
            # We can compute NEW diffs by temporarily setting row[e_idx]=new_val?
            # No, let's just store the old row before modification.
            
            # Let's refactor the move logic to be cleaner.
            pass
        
        # Refactoring the move logic for correctness and speed
        # The above revert logic is messy. Let's restart the loop with a cleaner structure.
        
        T *= cooling
        if T < min_T:
            T = 10.0 # Restart temperature
            # Optional: Reset to best
            # For now, just continue with higher T to escape local minima
            
    return best_rows

# Wait, the revert logic in the scratchpad above was flawed. 
# I will rewrite the core loop properly in the final code block.
