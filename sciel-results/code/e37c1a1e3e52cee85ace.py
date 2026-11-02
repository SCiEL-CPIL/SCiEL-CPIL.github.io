import time
import random
import math

def construct(n_rows, k):
    # n_rows=7, k=5 (so 6 elements per row)
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # 1. Generate initial valid solution using greedy
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = [x - a for a in row]
                # Check if any diff is in used
                valid = True
                for d in new_diffs:
                    if d in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for d in new_diffs:
                        used.add(d)
            rows.append(row)
        return rows

    best_rows = greedy_init()
    best_max = max(max(row) for row in best_rows)
    
    # 2. Simulated Annealing
    # State: rows
    # Cost: 10000 * num_duplicates + max_value
    
    def get_cost(rows):
        diff_counts = {}
        max_val = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    if row[j] > max_val:
                        max_val = row[j]
        
        n_dups = 0
        for count in diff_counts.values():
            if count > 1:
                n_dups += count - 1
        
        return 10000 * n_dups + max_val

    # Incremental update for speed
    # We will store current diff_counts and current max_val
    # But max_val is tricky to update incrementally if the max element moves.
    # Given small size, recalculating max_val is cheap. Recalculating dups is cheap too.
    # Let's just recalculate cost from scratch for simplicity first, 
    # but 105 diffs is very small. Recalculation is O(105). 
    # SA steps can be millions. 105 * 1M = 100M ops. Might be slow in Python.
    # Let's try to be smarter.
    
    # Actually, let's use the incremental approach for diff_counts.
    
    # Initialize diff_counts for best_rows
    def calc_diff_counts(rows):
        counts = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def get_n_dups(counts):
        n = 0
        for v in counts.values():
            if v > 1:
                n += v - 1
        return n

    current_rows = [row[:] for row in best_rows]
    current_counts = calc_diff_counts(current_rows)
    current_n_dups = get_n_dups(current_counts)
    current_max = max(max(row) for row in current_rows)
    current_cost = 10000 * current_n_dups + current_max
    
    best_cost = current_cost
    best_state = [row[:] for row in current_rows]
    
    T = 100.0
    T_min = 0.1
    cooling = 0.9999
    
    while time.time() < deadline:
        # Pick random move
        r_idx = random.randint(0, n_rows - 1)
        # Pick a non-zero index in the row
        # Indices 1..k
        i_idx = random.randint(1, k)
        
        old_val = current_rows[r_idx][i_idx]
        
        # Determine range for new_val
        # Must be > current_rows[r_idx][i_idx-1] and < current_rows[r_idx][i_idx+1] (if exists)
        lo = current_rows[r_idx][i_idx-1] + 1
        hi = current_rows[r_idx][i_idx+1] - 1 if i_idx < k else current_max + 100 # Allow growth? 
        # Actually, to minimize max, we should probably restrict hi to current_max or slightly above.
        # If we restrict to current_max, we can only decrease or keep same.
        # Let's allow hi to be current_max.
        hi = min(hi, current_max)
        
        if lo > hi:
            continue
            
        new_val = random.randint(lo, hi)
        if new_val == old_val:
            continue
            
        # Apply move
        # Remove old differences involving (r_idx, i_idx)
        row = current_rows[r_idx]
        for j in range(len(row)):
            if j == i_idx:
                continue
            if j < i_idx:
                d = old_val - row[j]
            else:
                d = row[j] - old_val
            current_counts[d] -= 1
            if current_counts[d] == 0:
                del current_counts[d]
                
        # Update row
        row[i_idx] = new_val
        
        # Add new differences
        for j in range(len(row)):
            if j == i_idx:
                continue
            if j < i_idx:
                d = new_val - row[j]
            else:
                d = row[j] - new_val
            current_counts[d] = current_counts.get(d, 0) + 1
            
        # Calculate new metrics
        new_n_dups = get_n_dups(current_counts)
        new_max = max(max(row) for row in current_rows)
        new_cost = 10000 * new_n_dups + new_max
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_n_dups = new_n_dups
            current_max = new_max
            current_cost = new_cost
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_state = [r[:] for r in current_rows]
                best_max = new_max
        else:
            # Reject: Undo move
            # Remove new differences
            for j in range(len(row)):
                if j == i_idx:
                    continue
                if j < i_idx:
                    d = new_val - row[j]
                else:
                    d = row[j] - new_val
                current_counts[d] -= 1
                if current_counts[d] == 0:
                    del current_counts[d]
                    
            # Restore row
            row[i_idx] = old_val
            
            # Add old differences
            for j in range(len(row)):
                if j == i_idx:
                    continue
                if j < i_idx:
                    d = old_val - row[j]
                else:
                    d = row[j] - old_val
                current_counts[d] = current_counts.get(d, 0) + 1
                
        T *= cooling
        if T < T_min:
            T = T_min

    # Return the best valid state found
    # Ensure the best state is valid (n_dups == 0)
    # If not, return the initial greedy one which is valid
    best_counts = calc_diff_counts(best_state)
    if get_n_dups(best_counts) == 0:
        return best_state
    else:
        return best_rows
