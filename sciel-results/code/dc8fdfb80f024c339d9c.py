import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def greedy_row(used_diffs):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    break
                x += 1
        return row

    initial_rows = []
    used_diffs = set()
    for _ in range(n_rows):
        row = greedy_row(used_diffs)
        initial_rows.append(row)
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    best_rows = [r[:] for r in initial_rows]
    best_scope = max(max(r) for r in best_rows)
    
    current_rows = [r[:] for r in initial_rows]
    
    diff_counts = {}
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
    def count_collisions(counts):
        return sum(c - 1 for c in counts.values() if c > 1)

    current_collisions = count_collisions(diff_counts)
    current_max = max(max(r) for r in current_rows)
    
    PENALTY = 1000
    
    def objective(max_val, collisions):
        return max_val + PENALTY * collisions

    current_obj = objective(current_max, current_collisions)
    
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        i = random.randint(1, k)
        old_val = row[i]
        
        low = row[i-1] + 1
        high = row[i+1] - 1 if i < k else current_max + 20
        
        if low >= high:
            continue
            
        new_val = random.randint(low, high)
        
        if new_val == old_val:
            continue
            
        diffs_to_remove = []
        diffs_to_add = []
        
        for j in range(i+1, len(row)):
            diffs_to_remove.append(row[j] - old_val)
            diffs_to_add.append(row[j] - new_val)
            
        for j in range(i):
            diffs_to_remove.append(old_val - row[j])
            diffs_to_add.append(new_val - row[j])
            
        # Incremental collision update
        delta_collisions = 0
        
        # Remove
        for d in diffs_to_remove:
            c = diff_counts.get(d, 0)
            if c == 0:
                # Should not happen if state is consistent
                continue
            if c == 2:
                delta_collisions -= 1
            elif c > 2:
                delta_collisions -= 1
            # if c == 1, no change to collision count
            
            # Update count for next removals or additions? 
            # We are simulating, so we should update a local view or apply to diff_counts?
            # If we apply to diff_counts, we must rollback on rejection.
            # Let's apply to diff_counts directly, and rollback if rejected.
            
            diff_counts[d] = c - 1
            if diff_counts[d] == 0:
                del diff_counts[d]
                
        # Add
        for d in diffs_to_add:
            c = diff_counts.get(d, 0)
            if c == 1:
                delta_collisions += 1
            elif c > 1:
                delta_collisions += 1
            # if c == 0, no change
            
            diff_counts[d] = c + 1
            
        new_collisions = current_collisions + delta_collisions
        
        new_row = row[:]
        new_row[i] = new_val
        new_row_max = new_row[-1]
        
        new_global_max = current_max
        # If the modified row was the max, and we reduced it, max might drop.
        # If we increased it, max might rise.
        # Simplest: recalculate max. n_rows is small.
        # Or: 
        # old_max_row_max = row[-1]
        # if new_row_max > new_global_max: new_global_max = new_row_max
        # elif old_max_row_max == current_max and new_row_max < current_max:
        #    new_global_max = max(max(r) for r in current_rows) # Recalc
        
        # For safety and simplicity, recalc max.
        new_global_max = 0
        for idx, r in enumerate(current_rows):
            if idx == r_idx:
                if new_row_max > new_global_max:
                    new_global_max = new_row_max
            else:
                if r[-1] > new_global_max:
                    new_global_max = r[-1]
                    
        new_obj = objective(new_global_max, new_collisions)
        
        if new_obj < current_obj or (T > T_min and random.random() < math.exp(-(new_obj - current_obj) / T)):
            current_rows[r_idx] = new_row
            current_collisions = new_collisions
            current_max = new_global_max
            current_obj = new_obj
            
            if new_collisions == 0 and new_global_max < best_scope:
                best_scope = new_global_max
                best_rows = [r[:] for r in current_rows]
        else:
            # Rollback diff_counts
            # Re-add removed
            for d in diffs_to_remove:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            # Remove added
            for d in diffs_to_add:
                c = diff_counts.get(d, 0)
                diff_counts[d] = c - 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            # current_collisions, current_max, current_obj unchanged
            
        T *= cooling
        if T < T_min:
            T = T_min

    return best_rows
