import time
import random
import math

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Greedy Initialization
    # used_diffs: set of all differences used so far
    # rows: list of lists
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        rows.append(row)
    
    # Initial best
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    # Precompute differences for each row to speed up updates
    # diffs[i] is a set of differences for row i
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    current_diffs = [get_diffs(r) for r in rows]
    
    # Helper to check if a new row value is valid
    # row_idx: index of the row being modified
    # pos: index in the row being modified
    # new_val: new value
    # returns: (is_valid, new_diffs_set)
    def check_move(row_idx, pos, new_val):
        # Check bounds within the row
        if pos > 0 and new_val <= rows[row_idx][pos-1]:
            return False, None
        if pos < k and new_val >= rows[row_idx][pos+1]:
            return False, None
        
        # Construct the new row temporarily to get its diffs
        # We only need to check against other rows
        # Let's compute the new diffs for this row
        new_row = rows[row_idx][:]
        new_row[pos] = new_val
        new_diffs = get_diffs(new_row)
        
        # Check against all other rows
        # Optimization: We can maintain a global set of diffs for all OTHER rows
        # But for simplicity in this step, let's just check intersection
        # Actually, maintaining a global set is better for speed.
        # Let's refactor to use a global set of diffs for all rows except current.
        
        # For now, simple check:
        for i in range(n_rows):
            if i == row_idx:
                continue
            if new_diffs & current_diffs[i]:
                return False, None
        return True, new_diffs

    # Better approach: Maintain `global_diffs` which is the union of diffs of all rows.
    # When checking a move on row `r`, we need to ensure `new_diffs_r` is disjoint from `global_diffs - old_diffs_r`.
    # Let's maintain `global_diffs` as a Counter or just a set, and handle removals/additions.
    # Since differences are unique across ALL rows, `global_diffs` is just a set.
    # But to check a move on row `r`, we need to know if `new_diffs_r` intersects with `global_diffs - old_diffs_r`.
    # This is equivalent to checking if `(new_diffs_r & global_diffs) - old_diffs_r` is empty?
    # No. `new_diffs_r` must be disjoint from `global_diffs - old_diffs_r`.
    # So, `new_diffs_r & (global_diffs - old_diffs_r)` must be empty.
    # This is true if and only if `new_diffs_r & global_diffs` is a subset of `old_diffs_r`.
    # Since `new_diffs_r` and `old_diffs_r` are sets of positive integers, and the row is strictly increasing,
    # the diffs are unique within the row.
    # So we just need to check: are any elements in `new_diffs_r` present in `global_diffs` but NOT in `old_diffs_r`?
    
    global_diffs = set()
    for d in current_diffs:
        global_diffs |= d
        
    def check_move_fast(row_idx, pos, new_val):
        if pos > 0 and new_val <= rows[row_idx][pos-1]:
            return False, None
        if pos < k and new_val >= rows[row_idx][pos+1]:
            return False, None
            
        new_row = rows[row_idx][:]
        new_row[pos] = new_val
        new_diffs = get_diffs(new_row)
        old_diffs = current_diffs[row_idx]
        
        # Check if any new diff is in global_diffs but was not in old_diffs
        # i.e., new_diffs & (global_diffs - old_diffs) must be empty
        # Or simpler: (new_diffs & global_diffs) - old_diffs is empty
        # Since old_diffs is a subset of global_diffs (currently),
        # we can just check:
        # for d in new_diffs:
        #     if d in global_diffs and d not in old_diffs:
        #         return False
        # This is O(M) check.
        
        # Even faster:
        # intersection = new_diffs & global_diffs
        # if intersection - old_diffs:
        #    return False
        
        intersection = new_diffs & global_diffs
        if intersection - old_diffs:
            return False, None
            
        return True, new_diffs

    def apply_move(row_idx, pos, new_val, new_diffs):
        old_val = rows[row_idx][pos]
        old_diffs = current_diffs[row_idx]
        
        # Update global_diffs
        # Remove old_diffs, add new_diffs
        # Note: global_diffs is a set.
        # We must be careful.
        # global_diffs = (global_diffs - old_diffs) | new_diffs
        # This works because old_diffs and new_diffs are disjoint from other rows' diffs.
        # And we already checked that new_diffs doesn't conflict with other rows.
        
        global_diffs.difference_update(old_diffs)
        global_diffs.update(new_diffs)
        
        # Update row and current_diffs
        rows[row_idx][pos] = new_val
        current_diffs[row_idx] = new_diffs

    def revert_move(row_idx, pos, old_val, old_diffs):
        new_val = rows[row_idx][pos]
        new_diffs = current_diffs[row_idx]
        
        global_diffs.difference_update(new_diffs)
        global_diffs.update(old_diffs)
        
        rows[row_idx][pos] = old_val
        current_diffs[row_idx] = old_diffs

    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # Heuristic for move selection:
    # Bias towards moving elements that are large or in rows with large max.
    
    while time.time() < deadline and T > min_T:
        # Select a row and position
        # Weight rows by their max element?
        # Let's just pick random row and pos for now.
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)  # pos 0 is fixed at 0
        
        old_val = rows[row_idx][pos]
        
        # Generate a new value
        # Try to move towards a random target or just random step
        # Let's try random step size
        # Step size should be proportional to current value?
        # Or just small integer step.
        # Let's try: new_val = old_val + random.randint(-10, 10)
        # But we must respect bounds.
        
        # Better: pick a random target in the valid range and interpolate?
        # Or just try a few random candidates.
        
        candidates = []
        # Try to decrease (to reduce max)
        if old_val > 1:
            candidates.append(old_val - random.randint(1, 5))
        # Try to increase (to allow others to decrease later?)
        if old_val < 200: # Arbitrary upper bound for exploration
            candidates.append(old_val + random.randint(1, 5))
        # Try random swap with another element in the same row? No, must keep increasing.
        
        # Also try to "jump" to a random position in the gap
        lower_bound = rows[row_idx][pos-1] + 1 if pos > 0 else 1
        upper_bound = rows[row_idx][pos+1] - 1 if pos < k else 200
        
        if lower_bound < upper_bound:
             candidates.append(random.randint(lower_bound, upper_bound))
        
        move_made = False
        for new_val in candidates:
            if new_val == old_val:
                continue
            if new_val < lower_bound or new_val > upper_bound:
                continue
                
            is_valid, new_diffs = check_move_fast(row_idx, pos, new_val)
            if not is_valid:
                continue
                
            # Calculate cost change
            # Cost is max element.
            # We need to know if this move changes the global max.
            # Current global max is best_max (or we can track current_max).
            # Let's track current_max.
            
            # For simplicity in SA, let's just use the max of the row if it's the max row,
            # or just check if new_val > current_max.
            # Actually, the cost is max(all elements).
            # If we change one element, the new global max is:
            # max( max(elements not in this row), max(elements in this row with new_val) )
            
            # To make this efficient, we can just calculate the new row's max and compare with global max excluding this row?
            # That's expensive.
            
            # Alternative: Just use the value of the moved element as a proxy? No.
            # Let's maintain `current_max`.
            # If we decrease an element that was the unique global max, current_max might drop.
            # If we increase an element, current_max might rise.
            
            # Let's just calculate the new global max by scanning?
            # N is small (4). K is small (8). 32 elements.
            # Scanning 32 integers is very fast.
            
            # Let's do it properly.
            # We need to update `rows` temporarily to calculate max?
            # No, we can just calculate:
            # new_row_max = max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:])
            # other_max = max(max(rows[i]) for i in range(n_rows) if i != row_idx)
            # new_global_max = max(new_row_max, other_max)
            
            # To avoid repeated max calculations for all rows, we can precompute max for each row.
            # row_maxes[i] = max(rows[i])
            # global_max = max(row_maxes)
            
            # Let's implement this optimization.
            
            # Re-initialize row_maxes outside the loop?
            # We need to update it.
            
            break # Only try the first valid candidate? Or try all?
            # Let's try all candidates and pick the best one (lowest cost).
            # Or just accept the first valid one in SA.
            # Let's stick to first valid for speed.
            
        # If no valid move was found in candidates, try a random step again?
        # Or just skip.
        
        # Let's restart the move generation logic to be cleaner.
        
        # --- Refined Move Generation ---
        # Pick row, pos
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        lower_bound = rows[row_idx][pos-1] + 1
        upper_bound = rows[row_idx][pos+1] - 1 if pos < k else 500 # Large upper bound
        
        if lower_bound >= upper_bound:
            continue
            
        # Generate 5 random candidates
        found_move = False
        for _ in range(5):
            new_val = random.randint(lower_bound, upper_bound)
            if new_val == rows[row_idx][pos]:
                continue
                
            is_valid, new_diffs = check_move_fast(row_idx, pos, new_val)
            if not is_valid:
                continue
                
            # Calculate new cost
            # We need current row maxes
            # Let's compute on the fly for simplicity first, optimize if slow.
            # n_rows=4, k=8 -> 32 items.
            
            # Temporarily update to get max? No.
            # new_row_max = max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:])
            # other_maxes = [max(rows[i]) for i in range(n_rows) if i != row_idx]
            # new_global_max = max(new_row_max, max(other_maxes) if other_maxes else 0)
            
            # This is O(N*K). 4*8=32 ops. Very fast.
            
            new_row_max = max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:])
            other_max = 0
            for i in range(n_rows):
                if i != row_idx:
                    if max(rows[i]) > other_max:
                        other_max = max(rows[i])
            
            new_global_max = max(new_row_max, other_max)
            
            old_global_max = best_max # Wait, we should track current_max, not best_max.
            # Let's track current_max.
            # We didn't define current_max in the loop yet.
            # Let's assume current_max is the max of the current state.
            
            # We need to initialize current_max.
            # Let's add that.
            
            # If we accept:
            #   apply_move
            #   update current_max = new_global_max
            #   if new_global_max < best_max: update best
            #   if new_global_max > old_global_max:
            #       delta = new_global_max - old_global_max
            #       if random() < exp(-delta/T): accept
            #   else: accept
            
            # We need `current_max` variable.
            
            break # Accept first valid move for this iteration of SA
            
    # The code above is getting messy with variable scope. Let's rewrite cleanly.

    # --- Clean Rewrite ---
    
    # Re-initialize state from greedy
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        rows.append(row)
        
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)
    current_max = best_max
    
    current_diffs = [get_diffs(r) for r in rows]
    global_diffs = set()
    for d in current_diffs:
        global_diffs |= d
        
    # Precompute row maxes
    row_maxes = [max(r) for r in rows]
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    no_improve_count = 0
    
    while time.time() < deadline and T > min_T:
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        lower_bound = rows[row_idx][pos-1] + 1
        upper_bound = rows[row_idx][pos+1] - 1 if pos < k else 500
        
        if lower_bound >= upper_bound:
            continue
            
        # Try a few random moves
        move_applied = False
        for _ in range(3):
            new_val = random.randint(lower_bound, upper_bound)
            if new_val == rows[row_idx][pos]:
                continue
                
            is_valid, new_diffs = check_move_fast(row_idx, pos, new_val)
            if not is_valid:
                continue
                
            # Calculate new global max
            new_row_max = max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:])
            # Update row_maxes temporarily for calculation?
            # Just calculate other max
            other_max = 0
            for i in range(n_rows):
                if i != row_idx:
                    if row_maxes[i] > other_max:
                        other_max = row_maxes[i]
            
            new_global_max = max(new_row_max, other_max)
            
            delta = new_global_max - current_max
            
            accept = False
            if delta < 0:
                accept = True
            else:
                if delta == 0:
                    accept = True
                else:
                    if random.random() < math.exp(-delta / T):
                        accept = True
            
            if accept:
                # Apply move
                old_val = rows[row_idx][pos]
                old_diffs = current_diffs[row_idx]
                old_row_max = row_maxes[row_idx]
                
                # Update global_diffs
                global_diffs.difference_update(old_diffs)
                global_diffs.update(new_diffs)
                
                # Update row
                rows[row_idx][pos] = new_val
                current_diffs[row_idx] = new_diffs
                row_maxes[row_idx] = new_row_max
                
                # Update current_max
                current_max = new_global_max
                
                # Update best
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                move_applied = True
                break # Accept first valid move
        
        if not move_applied:
             # Optional: try to restart if stuck?
             pass
             
        T *= cooling
        
    return best_rows

# The function get_diffs is defined inside construct in the draft, but needs to be accessible.
# Let's define it outside or as a nested function.
# In the clean rewrite, I used get_diffs. I need to make sure it's defined.

# Let's assemble the final code.

