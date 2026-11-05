import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for safety
    
    # 1. Greedy Initialization
    def greedy_solve():
        used = {} # diff -> count
        rows = []
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k+1):
                x = row[-1] + 1
                while True:
                    # Check if adding x creates any duplicate difference
                    # Diffs: x - row[i] for i in 0..j-1
                    conflict = False
                    for i in range(j):
                        d = x - row[i]
                        if d in used:
                            conflict = True
                            break
                    if not conflict:
                        break
                    x += 1
                row.append(x)
                for i in range(j):
                    d = x - row[i]
                    if d in used:
                        used[d] += 1
                    else:
                        used[d] = 1
        return rows
    
    rows = greedy_solve()
    
    # Verify validity and get scope
    def get_scope(rows):
        return max(row[-1] for row in rows)
    
    # Check if greedy solution is valid
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    best_rows = rows[:]
    best_scope = get_scope(rows)
    if is_valid(rows):
        # It is valid, so best_rows is valid
        pass
    else:
        # Should not happen with greedy, but if it does, we try to fix?
        # Greedy ensures distinctness by construction.
        pass

    # 2. Simulated Annealing
    # State: rows
    # Cost: penalty (conflicts) + lambda * scope
    # We want penalty 0.
    
    # Pre-compute diff counts for current rows
    diff_counts = {}
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in diff_counts:
                    diff_counts[d] += 1
                else:
                    diff_counts[d] = 1
                    
    def calc_penalty():
        p = 0
        for c in diff_counts.values():
            if c > 1:
                p += c - 1
        return p
        
    current_penalty = calc_penalty()
    current_scope = best_scope
    
    # SA Parameters
    T_start = 100.0
    T_end = 0.1
    alpha = 0.999
    steps = 100000
    T = T_start
    
    # Helper to update penalty and counts
    def add_diff(d):
        nonlocal current_penalty
        if d in diff_counts:
            c = diff_counts[d]
            if c == 1:
                current_penalty += 1
            else:
                current_penalty += 1
            diff_counts[d] = c + 1
        else:
            diff_counts[d] = 1
            
    def remove_diff(d):
        nonlocal current_penalty
        c = diff_counts[d]
        if c == 2:
            current_penalty -= 1
        elif c > 2:
            current_penalty -= 1
        # if c == 1, penalty doesn't change
        diff_counts[d] = c - 1
        if diff_counts[d] == 0:
            del diff_counts[d]
            
    # SA Loop
    for step in range(steps):
        if time.time() > deadline:
            break
            
        r_idx = random.randint(0, n_rows - 1)
        j = random.randint(1, k) # Index in row to change
        
        row = rows[r_idx]
        old_val = row[j]
        
        # Determine range for new value
        low = row[j-1] + 1
        if j < k:
            high = row[j+1] - 1
        else:
            # If last element, allow some expansion
            high = old_val + 30 
            if high < low:
                high = low + 20
                
        if low > high:
            continue
            
        new_val = random.randint(low, high)
        if new_val == old_val:
            continue
            
        # Calculate changes
        # Apply removals
        for i in range(j):
            remove_diff(old_val - row[i])
        for l in range(j+1, len(row)):
            remove_diff(row[l] - old_val)
            
        # Apply additions
        for i in range(j):
            add_diff(new_val - row[i])
        for l in range(j+1, len(row)):
            add_diff(row[l] - new_val)
            
        new_penalty = current_penalty
        
        # Calculate new scope
        if j == k:
            # Temporarily set row[j] to new_val
            row[j] = new_val
            new_scope = max(r[-1] for r in rows)
            row[j] = old_val # Revert
        else:
            new_scope = current_scope
            
        old_cost = current_penalty * 1000 + current_scope
        new_cost = new_penalty * 1000 + new_scope
        
        delta_cost = new_cost - old_cost
        
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            if T > 0:
                # Avoid overflow in exp
                if delta_cost / T < 700:
                    prob = math.exp(-delta_cost / T)
                    if random.random() < prob:
                        accept = True
            
        if accept:
            # Commit
            row[j] = new_val
            current_penalty = new_penalty
            current_scope = new_scope
            if current_penalty == 0 and current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        else:
            # Revert
            # Remove additions
            for i in range(j):
                remove_diff(new_val - row[i])
            for l in range(j+1, len(row)):
                remove_diff(row[l] - new_val)
            
            # Add removals
            for i in range(j):
                add_diff(old_val - row[i])
            for l in range(j+1, len(row)):
                add_diff(row[l] - old_val)
                
        T *= alpha
        
    # Return best valid solution
    # Ensure best_rows is valid
    # If best_rows is empty (shouldn't happen), return rows
    if not best_rows:
        return rows
        
    return best_rows
