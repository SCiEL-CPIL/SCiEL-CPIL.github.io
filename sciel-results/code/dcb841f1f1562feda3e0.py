import time
import random
import math

def construct(n_rows, k):
    # 1. Get initial bound from greedy
    greedy_rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
                x += 1
            else:
                x += 1
        greedy_rows.append(row)
    
    best_solution = [list(r) for r in greedy_rows]
    best_max = max(max(r) for r in best_solution)
    
    # Helper to calculate cost
    def get_cost(rows):
        used_diffs = set()
        conflicts = 0
        max_val = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used_diffs:
                        conflicts += 1
                    else:
                        used_diffs.add(d)
            max_val = max(max_val, max(r))
        # Weight conflicts to discourage them, but allow some exploration
        return max_val + 100 * conflicts

    current_rows = [list(r) for r in best_solution]
    current_cost = get_cost(current_rows)
    
    start_time = time.time()
    T = 500.0
    cooling_rate = 0.99
    min_T = 1.0
    
    while T > min_T and time.time() - start_time < 115:
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        # Pick a position j > 0
        j_idx = random.randint(1, k)
        
        # Determine valid range for new value
        low = current_rows[r_idx][j_idx - 1] + 1
        # Cap high to best_max + small buffer to keep search local
        high = min(current_rows[r_idx][j_idx + 1] - 1 if j_idx < k else best_max + 50, best_max + 50)
        
        if low > high:
            continue
            
        # Randomly choose a new value
        # Bias towards current value?
        # Uniform in [low, high]
        new_val = random.randint(low, high)
        
        # Store old value
        old_val = current_rows[r_idx][j_idx]
        if new_val == old_val:
            continue
            
        current_rows[r_idx][j_idx] = new_val
        
        new_cost = get_cost(current_rows)
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current_cost = new_cost
            # Update best if no conflicts
            # Extract conflicts from cost? No, we need to check.
            # We can just re-calculate conflicts for best update, or store it.
            # Since get_cost is called, let's just check if new_cost < best_max + 100 (implies 0 conflicts and lower max)
            # Actually, if conflicts > 0, cost >= 100.
            # If best_max is ~134, and we have 1 conflict, cost ~ 134+100=234.
            # If we have 0 conflicts and max 133, cost 133.
            # So if new_cost < best_max, it must be 0 conflicts and lower max.
            # If new_cost < best_max + 100, it might be 0 conflicts and same max, or 1 conflict and lower max.
            # We only care about 0 conflicts.
            # Let's explicitly check conflicts for best update.
            if new_cost < best_max:
                # This implies 0 conflicts and max_val < best_max
                best_max = new_cost
                best_solution = [list(r) for r in current_rows]
        else:
            # Revert
            current_rows[r_idx][j_idx] = old_val
            
        T *= cooling_rate
        
    return best_solution
