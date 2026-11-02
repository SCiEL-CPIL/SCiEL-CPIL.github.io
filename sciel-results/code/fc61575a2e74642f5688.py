import time
import random
import sys

# EVOLVE-BLOCK-START

def _compute_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    used = set()
    for row in rows:
        diffs = _compute_diffs(row)
        if used & diffs:
            return False
        used |= diffs
    return True

def _get_scope(rows):
    """Get the maximum entry in the rows."""
    return max(max(row) for row in rows)

def _greedy_construct(n_rows, k, rng):
    """Construct a valid set of rows using a greedy approach with randomization."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            # Look for the next mark
            # Start from the last mark + 1
            start_val = row[-1] + 1
            # Sometimes we might want to skip a few to explore different basins?
            # For greedy, we usually take the smallest valid.
            # Let's add a small random skip probability to diversify.
            skip = rng.randint(0, 2) 
            x = start_val + skip
            
            found = False
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
                x += 1
        rows.append(row)
    return rows

def _try_compact(rows, used):
    """
    Try to reduce the scope by shifting elements to the left.
    Returns True if any change was made.
    Note: 'used' must be the set of all diffs from 'rows'.
    This function modifies 'rows' and 'used' in place if successful.
    """
    changed = False
    # Iterate rows in reverse order of magnitude? Or just all rows.
    # We want to reduce the largest entries.
    # Sort rows by max value descending to prioritize compacting the largest ones?
    # Or just iterate all.
    
    # To be safe, we iterate multiple times until no changes occur.
    # But we must be careful with the 'used' set updates.
    
    # Let's try a single pass first.
    for r_idx in range(len(rows)):
        row = rows[r_idx]
        # Try to shift elements from right to left
        # We can't shift 0.
        for j in range(len(row) - 1, 0, -1):
            old_val = row[j]
            # Try to move to old_val - 1, then old_val - 2, etc.
            # But we must stay > row[j-1]
            min_val = row[j-1] + 1
            # We want to find the smallest valid v >= min_val
            # Actually, we want to reduce scope, so we try smaller values.
            # Start from min_val and go up? No, we want the smallest possible.
            # So we check if min_val is valid. If not, min_val+1, etc.
            # But we are currently at old_val. We want to go down.
            # So we check v from min_val to old_val - 1.
            
            # Optimization: We only need to check if the new diffs are free.
            # Diffs removed: old_val - row[i] for i < j, and row[l] - old_val for l > j
            # Diffs added: v - row[i] for i < j, and row[l] - v for l > j
            
            # This is expensive if done naively for every v.
            # Given k is small (7), row length is 8.
            # Diffs involved: j + (len(row)-1-j) = len(row)-1 = 7 diffs.
            # So for each v, we check 7 diffs.
            
            best_v = old_val
            for v in range(min_val, old_val):
                # Check if v is valid
                # 1. Check against used (excluding old diffs)
                # 2. Check internal consistency (v must be distinct from other row elements, which is guaranteed by range)
                
                # Get old diffs
                old_diffs = set()
                for i in range(j):
                    old_diffs.add(old_val - row[i])
                for l in range(j + 1, len(row)):
                    old_diffs.add(row[l] - old_val)
                
                # Get new diffs
                new_diffs = set()
                for i in range(j):
                    new_diffs.add(v - row[i])
                for l in range(j + 1, len(row)):
                    new_diffs.add(row[l] - v)
                
                # Check if new_diffs intersects with (used - old_diffs)
                # Note: 'used' contains old_diffs.
                # So we check if new_diffs & (used - old_diffs) is empty.
                # Also, new_diffs must not have duplicates (impossible as v is unique in row).
                
                # Efficient check:
                # If any d in new_diffs is in used and d is not in old_diffs, it's a conflict.
                conflict = False
                for d in new_diffs:
                    if d in used and d not in old_diffs:
                        conflict = True
                        break
                
                if not conflict:
                    # We found a valid smaller v.
                    # We want the smallest possible v to maximize compaction.
                    # So we can break here if we are scanning upwards from min_val?
                    # Yes, min_val is the smallest possible.
                    best_v = v
                    break
            
            if best_v != old_val:
                # Update row
                row[j] = best_v
                changed = True
                
                # Update used set
                # Remove old diffs
                for d in old_diffs:
                    used.remove(d)
                # Add new diffs
                used.update(new_diffs)
                
                # Note: If we change row[j], the diffs for indices > j change?
                # No, row[l] - v vs row[l] - old_val.
                # The set of diffs for the row changes.
                # The 'used' set update above handles this.
                
                # However, if we change row[j], does it affect the validity of row[j+1]?
                # The value row[j+1] is fixed. The diff row[j+1]-row[j] changes.
                # We already accounted for this in new_diffs/old_diffs.
                # But wait, if we shift row[j], the "min_val" constraint for row[j+1] might be violated?
                # No, we ensured v > row[j-1].
                # But we also need v < row[j+1].
                # In the loop `for v in range(min_val, old_val)`, we implicitly assume v < old_val.
                # Since old_val < row[j+1] (original valid row), and v < old_val, then v < row[j+1].
                # So the increasing order is preserved.
                
                # One subtlety: If we change row[j], the diffs for elements to the RIGHT of j
                # (i.e., row[l] - row[j] for l > j) are updated in 'used'.
                # This is correct.
                
                # But does this change affect the possibility of shifting row[j-1]?
                # Yes, because the "used" set has changed.
                # Our loop goes from right to left (j from len-1 down to 1).
                # So when we process j-1, the 'used' set reflects the change at j.
                # This is correct.
                
                # However, if we shift row[j] down, it might create a conflict with
                # differences from *other* rows that were previously fine?
                # The check `if d in used and d not in old_diffs` handles this.
                # 'used' contains diffs from all other rows.
                # So if the new diff d collides with a diff from another row, conflict is True.

    return changed

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    rng = random.Random(42)
    
    # 1. Initial Construction
    best_rows = None
    best_scope = float('inf')
    best_used = None
    
    # Run greedy a few times to get a good starting point
    for i in range(20):
        rows = _greedy_construct(n_rows, k, rng)
        if _is_valid(rows):
            scope = _get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                # Reconstruct used set for best_rows
                best_used = set()
                for r in best_rows:
                    best_used |= _compute_diffs(r)
                    
    if best_rows is None:
        # Fallback: simple greedy
        best_rows = _greedy_construct(n_rows, k, rng)
        best_scope = _get_scope(best_rows)
        best_used = set()
        for r in best_rows:
            best_used |= _compute_diffs(r)
            
    # 2. Compaction
    # Try to compact the best initial solution
    changed = True
    while changed and time.time() < deadline:
        changed = _try_compact(best_rows, best_used)
        if changed:
            best_scope = _get_scope(best_rows)
            
    # 3. Simulated Annealing
    # State: best_rows, best_used, best_scope
    # Move: Pick a row r, index j, try to change value.
    # We will focus on reducing the scope.
    # Temperature T, start high, decay.
    
    T = 10.0
    cooling = 0.995
    
    current_rows = [r[:] for r in best_rows]
    current_used = set(best_used)
    current_scope = best_scope
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        # Pick a move
        r_idx = rng.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Bias towards larger indices (end of row) as they contribute to scope
        # Weighted random choice
        weights = [1] * (k + 1)
        for j in range(1, k + 1):
            weights[j] = j + 1
            
        j = rng.choices(range(1, k + 1), weights=weights[1:])[0]
        
        old_val = row[j]
        
        # Generate new value
        # Try to decrease it
        # Range: (row[j-1], row[j+1] if j < k else old_val)
        min_v = row[j-1] + 1
        max_v = row[j+1] - 1 if j < k else old_val
        
        if min_v >= max_v:
            continue
            
        # Randomly pick a new value in [min_v, max_v]
        # Or try old_val - delta
        # Let's try a random value in the range to explore more
        new_val = rng.randint(min_v, max_v)
        
        if new_val == old_val:
            continue
            
        # Check validity
        # Diffs removed
        old_diffs = set()
        for i in range(j):
            old_diffs.add(old_val - row[i])
        for l in range(j + 1, k + 1):
            old_diffs.add(row[l] - old_val)
            
        # Diffs added
        new_diffs = set()
        for i in range(j):
            new_diffs.add(new_val - row[i])
        for l in range(j + 1, k + 1):
            new_diffs.add(row[l] - new_val)
            
        # Check conflict
        # New diffs must not be in (current_used - old_diffs)
        # And new_diffs must not have internal duplicates (guaranteed by strict increase)
        
        conflict = False
        for d in new_diffs:
            if d in current_used and d not in old_diffs:
                conflict = True
                break
                
        if conflict:
            continue
            
        # Calculate new scope
        new_max = max(row)
        # If we changed the max element, new_max might decrease
        # If we changed a non-max element, new_max stays same (unless we increased it, which we don't really want)
        # Actually, if we increased a non-max element, it might become the new max?
        # Unlikely if we bias towards decreasing, but possible.
        
        # Let's just compute max of the modified row
        # Since only row[j] changed:
        if new_val > old_val:
             # It might increase the max if j was the max or close to it
             potential_new_max = max(row[:j] + [new_val] + row[j+1:])
        else:
             # It might decrease the max if j was the max
             if j == k: # last element
                 potential_new_max = max(row[:j] + [new_val])
             else:
                 potential_new_max = max(row) # Max is elsewhere, or new_val is smaller than current max
        
        # Actually, simpler:
        temp_row = row[:]
        temp_row[j] = new_val
        new_scope = max(temp_row)
        
        # Acceptance criterion
        delta_scope = new_scope - current_scope
        
        if delta_scope < 0:
            # Improvement
            # Update state
            row[j] = new_val
            current_used -= old_diffs
            current_used |= new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
                best_used = set(current_used)
                
        else:
            # Worse or equal
            # Accept with probability exp(-delta_scope / T)
            prob = 2.718281828 ** (-delta_scope / T)
            if rng.random() < prob:
                row[j] = new_val
                current_used -= old_diffs
                current_used |= new_diffs
                current_scope = new_scope
        
        T *= cooling
        
        # Periodic compaction
        if rng.random() < 0.1:
             # Try to compact current state
             # Note: _try_compact modifies in place
             # We need to be careful if it fails? It only applies valid changes.
             # So it's safe.
             changed = _try_compact(current_rows, current_used)
             if changed:
                 current_scope = _get_scope(current_rows)
                 if current_scope < best_scope:
                     best_scope = current_scope
                     best_rows = [r[:] for r in current_rows]
                     best_used = set(current_used)
                     
    return best_rows

# EVOLVE-BLOCK-END
