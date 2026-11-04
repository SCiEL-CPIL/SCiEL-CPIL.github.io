import time
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115  # Leave 5s margin for safety
    
    # Lower bound for M
    min_M = int(n_rows * k * (k + 1) / 2)
    
    best_result = None
    best_M = float('inf')
    
    # We will use a randomized backtracking search for each M.
    
    for M in range(min_M, 200):
        if time.time() - start_time > time_limit:
            break
        
        # Try to construct n_rows rows for this M
        max_trials = 100 # Increase trials to find a solution faster if M is tight
        found = False
        
        for trial in range(max_trials):
            if time.time() - start_time > time_limit:
                break
                
            rng = random.Random(trial * 100 + M)
            used_diffs = 0
            rows = []
            success = True
            min_second_val = 1 # Symmetry breaking: a[0][1] <= a[1][1] ...
            
            for r in range(n_rows):
                row = [0]
                
                def backtrack(pos, used_diffs, min_second_val_constraint):
                    if pos == k + 1:
                        return used_diffs # Return the final mask on success
                    
                    if pos == 1:
                        start_v = max(min_second_val_constraint, 1)
                    else:
                        start_v = row[-1] + 1
                    
                    # Generate candidates
                    candidates = list(range(start_v, M + 1))
                    rng.shuffle(candidates)
                    
                    for v in candidates:
                        valid = True
                        diff_mask = 0
                        for u in row:
                            d = v - u
                            bit = 1 << d
                            if used_diffs & bit:
                                valid = False
                                break
                            if diff_mask & bit:
                                valid = False
                                break
                            diff_mask |= bit
                        
                        if not valid:
                            continue
                        
                        row.append(v)
                        used_diffs_new = used_diffs | diff_mask
                        
                        result = backtrack(pos + 1, used_diffs_new, min_second_val_constraint)
                        if result is not None:
                            return result
                        
                        row.pop()
                    
                    return None
                
                result_mask = backtrack(1, used_diffs, min_second_val)
                if result_mask is not None:
                    used_diffs = result_mask
                    rows.append(row)
                    min_second_val = row[1]
                else:
                    success = False
                    break
            
            if success:
                best_result = rows
                best_M = M
                found = True
                break # Found solution for this M, break trial loop
                
        if found:
            return best_result
            
    # If we didn't find a solution for M up to 199, return the best we have (which might be None)
    # But we must return a valid list of lists.
    # The greedy fallback from the original code is safer for validity.
    
    if best_result is None:
        # Fallback to greedy which always finds a solution (though M might be large)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                diff_mask = 0
                for u in row:
                    d = x - u
                    bit = 1 << d
                    if used & bit:
                        valid = False
                        break
                    if diff_mask & bit:
                        valid = False
                        break
                    diff_mask |= bit
                if valid:
                    row.append(x)
                    used |= diff_mask
            rows.append(row)
        best_result = rows

    return best_result
