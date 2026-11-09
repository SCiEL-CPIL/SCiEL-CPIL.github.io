import time
import random

def greedy_construct(n_rows, k):
    """
    Standard greedy construction to establish a baseline.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            diffs = {x - a for a in row}
            if not diffs & used:
                row.append(x)
                used.update(diffs)
        rows.append(row)
    return rows

def solve_sa(n_rows, k, initial_rows, initial_scope, deadline):
    """
    Simulated Annealing to minimize the maximum entry.
    """
    current_rows = [r[:] for r in initial_rows]
    current_max = initial_scope
    best_scope = initial_scope
    best_rows = [r[:] for r in initial_rows]
    
    def get_all_diffs(rows):
        diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    diffs.add(r[j] - r[i])
        return diffs
        
    used_diffs = get_all_diffs(current_rows)
    
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() < deadline - 5 and T > min_T:
        r_idx = random.randint(0, n_rows - 1)
        p_idx = random.randint(1, k)
        
        row = current_rows[r_idx]
        old_val = row[p_idx]
        
        # Random perturbation
        delta_range = 10
        new_val = old_val + random.randint(-delta_range, delta_range)
        
        min_val = row[p_idx - 1] + 1
        max_val = row[p_idx + 1] - 1 if p_idx < k else 10000
        
        if new_val < min_val:
            new_val = min_val
        if new_val > max_val:
            new_val = max_val
            
        if new_val == old_val:
            continue
            
        # Check validity
        removed = set()
        added = set()
        for i in range(len(row)):
            if i == p_idx:
                continue
            if i < p_idx:
                removed.add(old_val - row[i])
                added.add(new_val - row[i])
            else:
                removed.add(row[i] - old_val)
                added.add(row[i] - new_val)
        
        conflict = False
        for d in added:
            if d in used_diffs and d not in removed:
                conflict = True
                break
        
        if conflict:
            continue
        
        # Apply move
        row[p_idx] = new_val
        new_global_max = max(max(r) for r in current_rows)
        
        if new_global_max < current_max:
            current_max = new_global_max
            used_diffs.difference_update(removed)
            used_diffs.update(added)
            
            if current_max < best_scope:
                best_scope = current_max
                best_rows = [r[:] for r in current_rows]
        else:
            delta = new_global_max - current_max
            # Accept worse moves to escape local minima
            if random.random() < 1.0 / (1.0 + delta * delta):
                used_diffs.difference_update(removed)
                used_diffs.update(added)
                current_max = new_global_max
            else:
                # Revert
                row[p_idx] = old_val
        
        T *= cooling
        
    return best_rows, best_scope

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Greedy Baseline
    best_rows = greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Simulated Annealing
    best_rows, best_scope = solve_sa(n_rows, k, best_rows, best_scope, deadline)
    
    return best_rows
