import time
import random
import copy

def _greedy_construct(n_rows, k):
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
    return rows

def _get_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def construct(n_rows, k):
    deadline = time.time() + 145.0
    
    # 1. Initial Solution
    current_rows = _greedy_construct(n_rows, k)
    best_rows = copy.deepcopy(current_rows)
    
    def evaluate(rows):
        diff_counts = {}
        for row in rows:
            for d in _get_diffs(row):
                diff_counts[d] = diff_counts.get(d, 0) + 1
        conflicts = sum(c * (c - 1) // 2 for c in diff_counts.values())
        max_val = max(max(row) for row in rows)
        return conflicts, max_val

    current_conflicts, current_max = evaluate(current_rows)
    best_max = current_max
    best_conflicts = current_conflicts
    
    # 2. Simulated Annealing
    T = 100.0
    cooling = 0.999
    penalty = 10000.0
    
    def calc_cost(conflicts, max_val):
        return max_val + penalty * conflicts

    current_cost = calc_cost(current_conflicts, current_max)
    
    steps = 0
    last_improvement = 0
    
    while time.time() < deadline:
        # Pick random row and index
        r = random.randint(0, n_rows - 1)
        j = random.randint(1, k) # Indices 1..k
        
        v = current_rows[r][j]
        
        # Bounds
        lower = current_rows[r][j-1] + 1
        upper = current_rows[r][j+1] - 1 if j < k else current_max + 20
        
        if upper < lower:
            continue
            
        # Random perturbation
        # Sometimes make a bigger jump
        if random.random() < 0.1:
             delta = random.randint(-50, 50)
        else:
             delta = random.randint(-10, 10)
             
        new_v = v + delta
        
        if new_v < lower:
            new_v = lower
        if new_v > upper:
            new_v = upper
            
        if new_v == v:
            continue
            
        # Apply
        old_v = current_rows[r][j]
        current_rows[r][j] = new_v
        
        # Evaluate
        new_conflicts, new_max = evaluate(current_rows)
        new_cost = calc_cost(new_conflicts, new_max)
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < pow(2.718281828, -delta_cost / T):
            # Accept
            current_conflicts = new_conflicts
            current_max = new_max
            current_cost = new_cost
            
            if current_conflicts == 0 and current_max < best_max:
                best_max = current_max
                best_rows = copy.deepcopy(current_rows)
                best_conflicts = 0
                last_improvement = steps
        else:
            # Reject
            current_rows[r][j] = old_v
            
        T *= cooling
        steps += 1
        
        # If stuck for a long time, maybe increase T? Or restart?
        # Simple: just keep going.
        
    return best_rows
