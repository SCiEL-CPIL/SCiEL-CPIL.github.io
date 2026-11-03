import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit, leaving buffer
    
    # 1. Greedy initialization
    def greedy_init():
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
                    used.update(new_diffs)
            rows.append(row)
        return rows

    rows = greedy_init()
    
    # State variables
    def get_diffs(row):
        return [row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))]

    def get_conflict_penalty(diff_counts):
        # Sum of (count - 1) for all counts > 1
        return sum(c - 1 for c in diff_counts.values() if c > 1)

    # Initialize diff_counts
    diff_counts = {}
    for row in rows:
        for d in get_diffs(row):
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
    current_M = max(max(r) for r in rows)
    current_conflicts = get_conflict_penalty(diff_counts)
    current_cost = current_M + 100 * current_conflicts
    
    best_rows = [r[:] for r in rows]
    best_cost = current_cost
    best_M = current_M
    best_conflicts = current_conflicts
    
    # SA Parameters
    T = 50.0
    alpha = 0.999
    min_T = 0.1
    
    # Helper to update diff counts for a row
    def update_row_diffs(row, diff_counts):
        # Remove old
        for d in get_diffs(row):
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
        # Add new
        for d in get_diffs(row):
            diff_counts[d] = diff_counts.get(d, 0) + 1

    while time.time() < deadline:
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Pick a random index to modify (1 to k, index 0 is always 0)
        i_idx = random.randint(1, k)
        
        # Propose change: delta in [-5, 5]
        delta = random.randint(-5, 5)
        if delta == 0:
            continue
            
        new_val = row[i_idx] + delta
        
        # Check bounds: must be > row[i_idx-1] and < row[i_idx+1] (if exists)
        valid_move = True
        if new_val <= row[i_idx - 1]:
            valid_move = False
        if i_idx < k and new_val >= row[i_idx + 1]:
            valid_move = False
            
        if not valid_move:
            continue
            
        # Calculate cost change
        # 1. Update M
        old_max = max(row)
        new_row = row[:]
        new_row[i_idx] = new_val
        new_max = max(new_row)
        
        # Note: M is global max. If this row was the max holder, M might change.
        # Approximation: Just calculate new global M.
        # To be precise, we need to know if this row contained the global max.
        # For simplicity and speed, we can just compute new_M = max(max(r) for r in rows) after update?
        # That's O(N*K). N=4, K=8. Very fast.
        
        # 2. Update Conflicts
        # Remove old diffs, add new diffs
        old_diffs = get_diffs(row)
        new_diffs = get_diffs(new_row)
        
        # We need to calculate the change in conflict penalty.
        # Penalty = sum(max(0, c-1))
        # Let's just recalculate the penalty for the affected differences?
        # Or simpler: since N is small, just recalculate total conflicts?
        # No, maintaining a global dict is better.
        
        # Let's do the incremental update on a copy or undo logic.
        # Given small size, let's just compute the cost of the new state directly if we make the change.
        
        # Make change
        old_val = row[i_idx]
        row[i_idx] = new_val
        
        # Update diff_counts
        for d in old_diffs:
            diff_counts[d] -= 1
            if diff_counts[d] == 0: del diff_counts[d]
        for d in new_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
        new_conflicts = get_conflict_penalty(diff_counts)
        new_M = max(max(r) for r in rows)
        new_cost = new_M + 100 * new_conflicts
        
        delta_cost = new_cost - current_cost
        
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            if random.random() < math.exp(-delta_cost / T):
                accept = True
                
        if accept:
            current_cost = new_cost
            current_M = new_M
            current_conflicts = new_conflicts
            
            if new_conflicts == 0 and new_M < best_M:
                best_M = new_M
                best_cost = new_cost
                best_rows = [r[:] for r in rows]
                
        else:
            # Revert
            row[i_idx] = old_val
            for d in new_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0: del diff_counts[d]
            for d in old_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
        T *= alpha
        if T < min_T:
            # Restart temperature or randomize
            T = 10.0
            # Randomize one row to escape local opt
            r_idx = random.randint(0, n_rows - 1)
            rows[r_idx] = [0]
            x = 0
            # Generate a random valid row within scope 150
            while len(rows[r_idx]) < k + 1:
                x += random.randint(1, 5)
                if x > 150: break
                # Check if valid within row (strictly increasing)
                if len(rows[r_idx]) == 0 or x > rows[r_idx][-1]:
                    # Check global uniqueness? No, we allow conflicts during search.
                    # But we need strictly increasing.
                    rows[r_idx].append(x)
            # Update diff counts for this row
            # Remove old (which is now replaced, so we need to clear and re-add)
            # This is messy. Let's just recompute all diff_counts if we restart.
            diff_counts = {}
            for r in rows:
                for d in get_diffs(r):
                    diff_counts[d] = diff_counts.get(d, 0) + 1
            current_conflicts = get_conflict_penalty(diff_counts)
            current_M = max(max(r) for r in rows)
            current_cost = current_M + 100 * current_conflicts

    return best_rows
