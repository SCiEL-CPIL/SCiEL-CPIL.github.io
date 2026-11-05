import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 150s limit
    
    # Helper to get all positive differences in a row
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

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

    best_rows = greedy_init()
    best_scope = max(max(row) for row in best_rows)
    
    # Simulated Annealing
    def simulated_annealing():
        nonlocal best_rows, best_scope, start_time, time_limit
        
        # Max possible value for difference array
        # We need to estimate a max scope. 
        # Greedy scope is a good upper bound. 
        # Let's allocate a bit more for safety.
        max_val_bound = best_scope * 2 + 100
        
        # We will do multiple restarts
        for restart in range(10): # Max 10 restarts
            if time.time() - start_time > time_limit:
                break
            
            # Start from best solution
            rows = [row[:] for row in best_rows]
            
            # Calculate current sum
            current_sum = sum(sum(row) for row in rows)
            current_scope = max(max(row) for row in rows)
            
            # Initialize used_diffs array
            used_diffs = bytearray(max_val_bound + 1)
            
            # Mark all diffs from current rows
            valid = True
            for row in rows:
                for d in get_diffs(row):
                    if d > max_val_bound:
                        valid = False
                        break
                    if used_diffs[d]:
                        valid = False
                        break
                    used_diffs[d] = 1
                if not valid:
                    break
            
            if not valid:
                # This shouldn't happen if best_rows is valid
                # But just in case, fallback to greedy
                rows = greedy_init()
                current_sum = sum(sum(row) for row in rows)
                current_scope = max(max(row) for row in rows)
                used_diffs = bytearray(max_val_bound + 1)
                for row in rows:
                    for d in get_diffs(row):
                        if d <= max_val_bound:
                            used_diffs[d] = 1
                        else:
                            # Ignore for now, should not happen
                            pass
            
            temperature = 100.0
            cooling = 0.999
            min_temperature = 1.0
            
            while temperature > min_temperature and time.time() - start_time < time_limit:
                # Pick random row and position
                r = random.randint(0, n_rows - 1)
                i = random.randint(1, k) # 1 to k
                
                old_val = rows[r][i]
                
                # Determine valid range for new value
                low = rows[r][i-1] + 1
                high = rows[r][i+1] - 1 if i < k else max_val_bound
                
                if low >= high:
                    temperature *= cooling
                    continue
                
                # Pick new value
                new_val = random.randint(low, high)
                
                if new_val == old_val:
                    temperature *= cooling
                    continue
                
                # Remove old diffs
                old_diffs = []
                for j in range(i):
                    d = old_val - rows[r][j]
                    if d <= max_val_bound:
                        old_diffs.append(d)
                        used_diffs[d] = 0
                for j in range(i+1, len(rows[r])):
                    d = rows[r][j] - old_val
                    if d <= max_val_bound:
                        old_diffs.append(d)
                        used_diffs[d] = 0
                
                # Check new diffs
                new_diffs = []
                valid = True
                for j in range(i):
                    d = new_val - rows[r][j]
                    if d > max_val_bound:
                        valid = False
                        break
                    if used_diffs[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    for j in range(i+1, len(rows[r])):
                        d = rows[r][j] - new_val
                        if d > max_val_bound:
                            valid = False
                            break
                        if used_diffs[d]:
                            valid = False
                            break
                        new_diffs.append(d)
                
                if not valid:
                    # Restore old diffs
                    for d in old_diffs:
                        used_diffs[d] = 1
                    temperature *= cooling
                    continue
                
                # Calculate new sum
                delta_sum = new_val - old_val
                new_sum = current_sum + delta_sum
                
                # Check new scope
                new_row_max = max(rows[r][:i] + [new_val] + rows[r][i+1:])
                
                # Global max
                new_scope = 0
                for idx, row in enumerate(rows):
                    if idx == r:
                        m = new_row_max
                    else:
                        m = max(row)
                    if m > new_scope:
                        new_scope = m
                
                # Acceptance criterion
                accepted = False
                
                # If we find a better scope, always accept
                if new_scope < best_scope:
                    accepted = True
                    best_scope = new_scope
                    best_rows = [row[:] for row in rows]
                else:
                    # Use Metropolis criterion on sum
                    # We allow worsening of sum to escape local minima
                    # But we penalize worsening of scope implicitly by sum
                    if new_sum <= current_sum:
                        accepted = True
                    else:
                        delta = new_sum - current_sum
                        # Scale temperature by problem size?
                        prob = math.exp(-delta / temperature)
                        if random.random() < prob:
                            accepted = True
                
                if accepted:
                    # Update state
                    for d in new_diffs:
                        used_diffs[d] = 1
                    rows[r][i] = new_val
                    current_sum = new_sum
                else:
                    # Restore old diffs
                    for d in old_diffs:
                        used_diffs[d] = 1
                
                temperature *= cooling
    
    simulated_annealing()
    
    return best_rows
