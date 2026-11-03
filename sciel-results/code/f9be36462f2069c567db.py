import random
import math
import time

def construct(n_rows, k):
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
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    def get_max_val(rws):
        return max(r[-1] for r in rws)

    best_max = float('inf')
    best_rows = None
    start_time = time.time()
    time_limit = 115.0
    
    restart_idx = 0
    while time.time() - start_time < time_limit:
        restart_idx += 1
        rows, used = greedy_init()
        
        # Perturb initial state to escape greedy's specific local minimum
        # Randomly swap some values? No, must maintain increasing order and validity.
        # Just rely on SA.
        
        current_max = get_max_val(rows)
        local_best_rows = [r[:] for r in rows]
        local_best_max = current_max
        
        T = 80.0
        cooling = 0.9995
        iterations = 100000
        random.seed(12345 + restart_idx * 7919)
        
        for _ in range(iterations):
            if time.time() - start_time > time_limit:
                break
            if T < 0.05:
                break
                
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, k)
            
            v = rows[r][j]
            min_v = rows[r][j-1] + 1
            # Allow expansion
            max_v = max(current_max + 15, v + 10)
            
            v_prime = random.randint(min_v, max_v)
            if v_prime == v:
                continue
                
            old_diffs = set()
            new_diffs = set()
            # Efficient diff calculation
            # Since k is small, this is fine
            for idx, a in enumerate(rows[r]):
                if idx != j:
                    old_diffs.add(abs(v - a))
                    new_diffs.add(abs(v_prime - a))
                    
            # Check collision
            valid = True
            for d in new_diffs:
                if d in used and d not in old_diffs:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Apply move
            old_max = current_max
            rows[r][j] = v_prime
            used.difference_update(old_diffs)
            used.update(new_diffs)
            
            new_max = get_max_val(rows)
            current_max = new_max
            
            delta = new_max - old_max
            
            if delta <= 0:
                if new_max < local_best_max:
                    local_best_max = new_max
                    local_best_rows = [r[:] for r in rows]
            else:
                # Metropolis acceptance
                # Optimization: if delta is very large, reject immediately
                if delta < T * 10: 
                    if random.random() < math.exp(-delta / T):
                        pass # Accept
                    else:
                        # Reject, revert
                        rows[r][j] = v
                        used.difference_update(new_diffs)
                        used.update(old_diffs)
                        current_max = old_max
                else:
                    # Reject, revert
                    rows[r][j] = v
                    used.difference_update(new_diffs)
                    used.update(old_diffs)
                    current_max = old_max
                    
            T *= cooling
            
        if local_best_max < best_max:
            best_max = local_best_max
            best_rows = local_best_rows
            
        if best_max <= 112:
            break

    return best_rows
