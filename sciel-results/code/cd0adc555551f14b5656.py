import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115  # Leave some margin for overhead
    
    # Helper to check if a row is valid given used diffs
    def get_row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    # 1. Initial Greedy Solution
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows, used

    rows, used_diffs = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    # SA Parameters
    T = best_scope * 0.1
    T_min = 1e-5
    alpha = 0.999
    
    # Precompute row diffs for quick update? 
    # It's easier to just recalculate the affected diffs.
    
    def get_used_diffs_except_row(rows, r_idx, row):
        # This is O(N^2) if we do it naively. 
        # Better to maintain a global set and subtract/add.
        pass

    # We will maintain `used_diffs` as a global set.
    # When we change a row, we remove its old diffs, add new ones, and check for conflicts.
    
    # Re-initialize used_diffs properly from rows
    used_diffs = set()
    for r in rows:
        used_diffs |= get_row_diffs(r)
        
    current_scope = max(max(r) for r in rows)
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick a row to modify
        # Bias towards rows with larger max values
        max_val = max(max(r) for r in rows)
        candidate_rows = [i for i, r in enumerate(rows) if max(r) == max_val]
        if not candidate_rows:
            candidate_rows = list(range(n_rows))
        
        r_idx = random.choice(candidate_rows)
        row = rows[r_idx]
        
        # Pick an index to change. 
        # Changing the last element is most likely to reduce scope.
        # But sometimes we need to change earlier elements.
        # Let's pick a random index j in 1..k
        j = random.randint(1, k)
        
        old_val = row[j]
        
        # Determine range for new value
        # Must be > row[j-1] and < row[j+1] (if j < k)
        lower = row[j-1] + 1
        upper = row[j+1] - 1 if j < k else current_scope + 10 # Allow expansion if needed to escape
        
        if lower > upper:
            T *= alpha
            continue
            
        # Generate candidate x
        # To minimize scope, we want smaller x. 
        # But if we are stuck, we might need larger x.
        # Let's pick a random x in [lower, upper]
        # Or, if we want to reduce scope, pick x < old_val?
        # Let's try a mix: 50% chance to try smaller, 50% random in range.
        if random.random() < 0.5 and old_val > lower:
             # Try to reduce
             x = random.randint(lower, old_val - 1)
        else:
             x = random.randint(lower, upper)
             
        if x == old_val:
            continue
            
        # Check validity
        # Remove old diffs involving row[j]
        old_diffs_to_remove = set()
        for i in range(j):
            old_diffs_to_remove.add(old_val - row[i])
        for i in range(j+1, k+1):
            old_diffs_to_remove.add(row[i] - old_val)
            
        # Calculate new diffs involving x
        new_diffs_to_add = set()
        for i in range(j):
            new_diffs_to_add.add(x - row[i])
        for i in range(j+1, k+1):
            new_diffs_to_add.add(row[i] - x)
            
        # Check if new_diffs_to_add conflicts with (used_diffs - old_diffs_to_remove)
        # i.e., new_diffs_to_add & (used_diffs - old_diffs_to_remove) must be empty
        # This is equivalent to: new_diffs_to_add & used_diffs must be a subset of old_diffs_to_remove
        # But simpler: just check intersection with used_diffs, and ensure any intersection is in old_diffs_to_remove
        
        # Efficient check:
        # If new_diffs_to_add has any element not in used_diffs, it's fine.
        # If it has an element in used_diffs, that element MUST be in old_diffs_to_remove.
        
        conflict = False
        for d in new_diffs_to_add:
            if d in used_diffs:
                if d not in old_diffs_to_remove:
                    conflict = True
                    break
        
        if conflict:
            T *= alpha
            continue
            
        # Valid move
        # Update state
        used_diffs -= old_diffs_to_remove
        used_diffs |= new_diffs_to_add
        
        row[j] = x
        
        new_scope = max(max(r) for r in rows)
        
        # Acceptance
        if new_scope < current_scope:
            # Always accept improvement
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        else:
            # Deterioration or equal
            # Accept with probability exp(-delta / T)
            # Delta is increase in scope
            delta = new_scope - current_scope
            if delta > 0:
                prob = 2.718281828 ** (-delta / T)
                if random.random() < prob:
                    current_scope = new_scope
                else:
                    # Revert
                    row[j] = old_val
                    used_diffs -= new_diffs_to_add
                    used_diffs |= old_diffs_to_remove
            else:
                # Equal scope, accept to explore
                current_scope = new_scope
                
        T *= alpha
        
    return best_rows
