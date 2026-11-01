import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Generate initial valid solution using greedy
    rows = greedy_construct(n_rows, k)
    if not rows:
        return []
        
    current_scope = max(max(row) for row in rows)
    current_invalid = 0
    
    best_rows = [row[:] for row in rows]
    best_scope = current_scope
    best_invalid = 0
    
    # Data structure for quick difference counting
    diff_counts = {}
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
    # Simulated Annealing
    T = 10.0 # Initial temperature
    cooling_rate = 0.9995
    min_T = 0.01
    steps_since_improvement = 0
    max_steps_no_improvement = 10000
    
    # Helper to calculate scope quickly
    def get_scope(rows):
        return max(max(row) for row in rows)

    while time.time() < deadline:
        if T < min_T or steps_since_improvement > max_steps_no_improvement:
            # Restart: Perturb best solution
            rows = [row[:] for row in best_rows]
            # Apply some random valid moves to perturb
            for _ in range(50):
                r = random.randint(0, n_rows - 1)
                i = random.randint(1, k)
                old_val = rows[r][i]
                lo = rows[r][i-1] + 1
                hi = rows[r][i+1] - 1 if i < k else old_val + 20
                if lo <= hi:
                    new_val = random.randint(lo, hi)
                    if new_val != old_val:
                        # Apply move without cost check, just for perturbation
                        # This might make it invalid, which is fine for restart
                        # We just need to update state
                        # For simplicity, just rebuild diff_counts after perturbation
                        rows[r][i] = new_val
            
            # Rebuild diff_counts and state
            diff_counts = {}
            for row in rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        diff_counts[d] = diff_counts.get(d, 0) + 1
            current_scope = get_scope(rows)
            current_invalid = sum(1 for c in diff_counts.values() if c > 1)
            
            T = 10.0
            steps_since_improvement = 0
            
        # Pick a random move
        r = random.randint(0, n_rows - 1)
        i = random.randint(1, k)
        
        old_val = rows[r][i]
        
        lo = rows[r][i-1] + 1
        hi = rows[r][i+1] - 1 if i < k else old_val + 50
        
        if lo > hi:
            T *= cooling_rate
            continue
            
        new_val = random.randint(lo, hi)
        if new_val == old_val:
            T *= cooling_rate
            continue
            
        # Calculate delta cost
        old_diffs = set()
        for j in range(i):
            old_diffs.add(old_val - rows[r][j])
        for j in range(i+1, len(rows[r])):
            old_diffs.add(rows[r][j] - old_val)
            
        new_diffs = set()
        for j in range(i):
            new_diffs.add(new_val - rows[r][j])
        for j in range(i+1, len(rows[r])):
            new_diffs.add(rows[r][j] - new_val)
            
        affected = old_diffs.union(new_diffs)
        
        delta_invalid = 0
        for d in affected:
            old_c = diff_counts.get(d, 0)
            c_old_local = 1 if d in old_diffs else 0
            c_new_local = 1 if d in new_diffs else 0
            new_c = old_c - c_old_local + c_new_local
            
            old_invalid_contrib = 1 if old_c > 1 else 0
            new_invalid_contrib = 1 if new_c > 1 else 0
            delta_invalid += new_invalid_contrib - old_invalid_contrib
            
        # Apply move to calculate new scope
        rows[r][i] = new_val
        new_scope = get_scope(rows)
        delta_scope = new_scope - current_scope
        
        delta_cost = delta_invalid * 10000 + delta_scope
        
        # Acceptance
        accepted = False
        if delta_cost < 0:
            accepted = True
        else:
            if T > 0:
                prob = pow(2.718281828, -delta_cost / T)
                if random.random() < prob:
                    accepted = True
        
        if not accepted:
            # Revert
            rows[r][i] = old_val
            T *= cooling_rate
            continue
        
        # Update state
        for d in old_diffs:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
        for d in new_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
        current_invalid += delta_invalid
        current_scope = new_scope
        
        # Update best
        if current_invalid == 0 and current_scope < best_scope:
            best_scope = current_scope
            best_rows = [row[:] for row in rows]
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
            
        T *= cooling_rate

    return best_rows

def greedy_construct(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
