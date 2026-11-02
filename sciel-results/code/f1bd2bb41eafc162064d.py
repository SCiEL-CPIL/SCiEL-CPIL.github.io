import random
import time
from collections import Counter

def construct(n_rows, k):
    # 1. Greedy initialization
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
                used |= new_diffs
        rows.append(row)
        
    def get_row_diffs(row):
        return [row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))]
    
    current_max = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    T = 50.0
    T_min = 0.01
    alpha = 0.9995
    penalty_weight = 1000
    
    cur_max = current_max
    cur_conflicts = 0
    cur_cost = cur_max + penalty_weight * cur_conflicts
    
    used_counts = Counter()
    for r in rows:
        for d in get_row_diffs(r):
            used_counts[d] += 1
            
    while time.time() < deadline:
        if T < T_min:
            break
            
        r_idx = random.randint(0, n_rows - 1)
        e_idx = random.randint(1, k)
        
        old_val = rows[r_idx][e_idx]
        
        low = rows[r_idx][e_idx-1] + 1
        high = rows[r_idx][e_idx+1] - 1 if e_idx < k else cur_max + 50
        
        if low > high:
            continue
            
        # Bias towards lower values
        if random.random() < 0.7:
            upper_bound = min(high, old_val)
            if low <= upper_bound:
                new_val = random.randint(low, upper_bound)
            else:
                new_val = low
        else:
            new_val = random.randint(low, high)
            
        if new_val == old_val:
            continue
            
        old_affected = []
        for i in range(e_idx):
            old_affected.append(old_val - rows[r_idx][i])
        for j in range(e_idx + 1, len(rows[r_idx])):
            old_affected.append(rows[r_idx][j] - old_val)
            
        new_affected = []
        for i in range(e_idx):
            new_affected.append(new_val - rows[r_idx][i])
        for j in range(e_idx + 1, len(rows[r_idx])):
            new_affected.append(rows[r_idx][j] - new_val)
            
        # Apply
        for d in old_affected:
            used_counts[d] -= 1
            if used_counts[d] == 0:
                del used_counts[d]
                
        for d in new_affected:
            used_counts[d] += 1
            
        rows[r_idx][e_idx] = new_val
        
        new_max = max(max(r) for r in rows)
        new_conflicts = sum(1 for c in used_counts.values() if c > 1)
        
        new_cost = new_max + penalty_weight * new_conflicts
        delta = new_cost - cur_cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            cur_max = new_max
            cur_conflicts = new_conflicts
            cur_cost = new_cost
            
            if new_conflicts == 0 and new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in rows]
        else:
            # Revert
            rows[r_idx][e_idx] = old_val
            
            for d in old_affected:
                used_counts[d] += 1
                
            for d in new_affected:
                used_counts[d] -= 1
                if used_counts[d] == 0:
                    del used_counts[d]
            
        T *= alpha
        
    return best_rows
