import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Helper to calculate cost
    def get_cost(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.append(row[j] - row[i])
        
        # Count duplicates
        seen = {}
        dup_count = 0
        for d in diffs:
            if d in seen:
                dup_count += 1
            else:
                seen[d] = 1
        
        scope = max(max(row) for row in rows)
        # Heavily penalize duplicates to drive towards validity
        return scope + 100 * dup_count

    # Helper to get scope of a valid solution
    def get_scope(rows):
        return max(max(row) for row in rows)

    # Greedy initialization
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
                    used |= new_diffs
            rows.append(row)
        return rows

    # Initial state
    best_rows = greedy_init()
    best_cost = get_cost(best_rows)
    current_rows = [row[:] for row in best_rows]
    current_cost = best_cost
    
    # Simulated Annealing parameters
    T = 100.0
    cooling = 0.9999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        # Pick an index in the row (not 0, as it's fixed to 0)
        i_idx = random.randint(1, k)
        
        old_val = current_rows[r_idx][i_idx]
        # Random delta
        delta = random.randint(-5, 5)
        new_val = old_val + delta
        
        # Check structural validity (strictly increasing)
        if new_val <= 0:
            continue
            
        if i_idx > 0 and new_val <= current_rows[r_idx][i_idx - 1]:
            continue
        if i_idx < k and new_val >= current_rows[r_idx][i_idx + 1]:
            continue
            
        # Apply move
        current_rows[r_idx][i_idx] = new_val
        
        # Calculate new cost
        new_cost = get_cost(current_rows)
        
        # Acceptance criterion
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [row[:] for row in current_rows]
        else:
            # Revert
            current_rows[r_idx][i_idx] = old_val
            
        T *= cooling

    # Return the best valid solution found
    # Check if best_rows is valid
    def is_valid(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.append(row[j] - row[i])
        return len(diffs) == len(set(diffs))
        
    if is_valid(best_rows):
        return best_rows
    
    # Fallback: return the greedy solution which is always valid
    return greedy_init()
