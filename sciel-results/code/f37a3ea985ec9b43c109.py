import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0 # Safety margin
    
    row_size = k + 1
    diffs_per_row = row_size * (row_size - 1) // 2
    total_diffs = n_rows * diffs_per_row
    lb = total_diffs
    
    def get_metrics(rows):
        used = set()
        conflicts = 0
        max_val = 0
        for r in rows:
            # Check row validity (strictly increasing)
            for i in range(1, len(r)):
                if r[i] <= r[i-1]:
                    return float('inf'), float('inf')
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
            max_val = max(max_val, r[-1])
        return max_val, conflicts

    # 1. Greedy Initial Solution
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_size:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_max, best_conf = get_metrics(best_rows)
    
    # 2. Simulated Annealing
    # Temperature starts high, cools down.
    # Objective: Minimize (max_val, conflicts)
    # We want conflicts == 0.
    
    current_rows = [r[:] for r in best_rows]
    current_max, current_conf = get_metrics(current_rows)
    
    T = 100.0
    cooling = 0.99
    
    # Time limit for SA: 30 seconds
    sa_deadline = start_time + 30.0
    
    while time.time() < sa_deadline and T > 0.1:
        # Random Move
        r_idx = random.randint(0, n_rows - 1)
        j_idx = random.randint(1, row_size - 1) # Don't change 0
        
        old_val = current_rows[r_idx][j_idx]
        
        # Propose new value
        # Try to stay within current bounds or slightly expand
        max_bound = max(current_max, lb)
        new_val = random.randint(1, max_bound)
        
        # Ensure row remains strictly increasing
        if j_idx > 0 and new_val <= current_rows[r_idx][j_idx-1]:
            continue
        if j_idx < row_size - 1 and new_val >= current_rows[r_idx][j_idx+1]:
            continue
            
        # Apply move
        current_rows[r_idx][j_idx] = new_val
        
        new_max, new_conf = get_metrics(current_rows)
        
        # Delta in objective
        # We prioritize reducing conflicts, then max_val
        # Score = conflicts * 1000 + max_val
        old_score = current_conf * 1000 + current_max
        new_score = new_conf * 1000 + new_max
        delta = new_score - old_score
        
        if delta < 0:
            # Accept
            current_max, current_conf = new_max, new_conf
        else:
            # Accept with probability
            p = math.exp(-delta / T)
            if random.random() < p:
                current_max, current_conf = new_max, new_conf
            else:
                # Revert
                current_rows[r_idx][j_idx] = old_val
        
        T *= cooling
        
        # Update best
        if current_conf == 0:
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
        
        # If we found a solution with M close to LB, we can stop SA
        if best_max == lb:
            break

    # 3. Backtracking for M < best_max
    # We try to find a solution with M < best_max
    # Iterate M from lb to best_max - 1
    
    def solve_for_M(M):
        used_diffs = set()
        current_rows = []
        
        def backtrack(row_idx, current_row, x):
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                return [r[:] for r in current_rows]
            
            if len(current_row) == row_size:
                return backtrack(row_idx + 1, [0], 0)
            
            # Try next values
            # Pruning: If x + (row_size - len(current_row)) > M, we can't fit more
            # Actually, the gaps must be at least 1.
            min_needed = row_size - len(current_row)
            if x + min_needed > M:
                return None
                
            for next_x in range(x + 1, M + 1):
                new_diffs = {next_x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                current_row.append(next_x)
                used_diffs.update(new_diffs)
                
                result = backtrack(row_idx, current_row, next_x)
                if result is not None:
                    return result
                
                current_row.pop()
                used_diffs.difference_update(new_diffs)
                
            return None
        
        return backtrack(0, [0], 0)

    # Search for smaller M
    # We have best_max from SA.
    # Try M from lb to best_max - 1
    # Give each M a small time slice, e.g., 2 seconds
    # Total time remaining: deadline - time.time()
    
    time_left = deadline - time.time()
    if time_left > 5:
        m_range = range(lb, best_max)
        # If range is large, we might not check all.
        # But lb to best_max is likely small (e.g., 112 to 134 is 22 values).
        # 22 * 2s = 44s.
        # If time_left is 80s, we can do it.
        
        for M in m_range:
            if time.time() > deadline - 2:
                break
            
            # Run solver with a strict internal deadline
            # The solver uses `deadline` global, but we want to slice it.
            # We can modify the solver to accept a deadline.
            
            # For simplicity, we just run it. If it's slow, it will hit the global deadline.
            # To prevent one M from eating all time, we can't easily interrupt the recursive call 
            # without passing a deadline.
            
            # Let's pass a deadline to the solver.
            # Refactor solve_for_M to take a deadline.
            
            pass # Placeholder, will implement below

    return best_rows
