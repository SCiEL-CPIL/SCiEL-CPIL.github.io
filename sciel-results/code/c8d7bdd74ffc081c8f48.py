import random
import math
import time

def construct(n_rows, k):
    row_len = k + 1
    
    def greedy_solution():
        rows = [[0] for _ in range(n_rows)]
        used = set()
        for step in range(n_rows * k):
            # Heuristic: pick row with min length, then min last val
            target_row_idx = -1
            min_len = row_len
            min_last_val = float('inf')
            for i in range(n_rows):
                if len(rows[i]) < min_len:
                    min_len = len(rows[i])
                    target_row_idx = i
                    min_last_val = rows[i][-1]
                elif len(rows[i]) == min_len:
                    if rows[i][-1] < min_last_val:
                        min_last_val = rows[i][-1]
                        target_row_idx = i
            
            if target_row_idx == -1:
                break
            i = target_row_idx
            current_row = rows[i]
            x = current_row[-1] + 1
            while True:
                new_diffs = {x - v for v in current_row}
                if not (new_diffs & used):
                    current_row.append(x)
                    used.update(new_diffs)
                    break
                x += 1
        return rows, max(max(r) for r in rows)

    start_time = time.time()
    timeout = 110
    
    baseline_rows, baseline_cost = greedy_solution()
    best_solution = baseline_rows
    best_cost = baseline_cost
    
    beam_width = 5000 
    max_depth = n_rows * row_len
    
    initial_rows = [[0] for _ in range(n_rows)]
    initial_used = set()
    beam = [(0, initial_rows, initial_used)]
    
    for step in range(n_rows * k):
        if time.time() - start_time > timeout:
            break
            
        next_beam = []
        
        for cost, rows, used_diffs in beam:
            min_len = row_len
            target_row_idx = -1
            min_last_val = float('inf')
            
            for i in range(n_rows):
                if len(rows[i]) < min_len:
                    min_len = len(rows[i])
                    target_row_idx = i
                    min_last_val = rows[i][-1]
                elif len(rows[i]) == min_len:
                    if rows[i][-1] < min_last_val:
                        min_last_val = rows[i][-1]
                        target_row_idx = i
                        
            if target_row_idx == -1:
                current_max = max(max(r) for r in rows)
                if current_max < best_cost:
                    best_cost = current_max
                    best_solution = [r[:] for r in rows]
                continue

            i = target_row_idx
            current_row = rows[i]
            
            # Precompute blocked values for this row
            # x is invalid if x - v is in used_diffs for any v in current_row
            # => x is in {u + v for u in used_diffs, v in current_row}
            blocked = set()
            for v in current_row:
                for u in used_diffs:
                    blocked.add(u + v)
            
            limit_val = best_cost - 1
            x = current_row[-1] + 1
            
            candidates = []
            while x <= limit_val:
                if x not in blocked:
                    candidates.append(x)
                    if len(candidates) >= 10: # Reduced candidates to allow more states in beam
                        break
                x += 1
                
            if not candidates:
                continue

            for new_val in candidates:
                new_rows = [r[:] for r in rows]
                new_rows[i].append(new_val)
                
                new_used_diffs = set(used_diffs)
                for v in new_rows[i]:
                    if v != new_val:
                        new_used_diffs.add(new_val - v)
                        
                new_cost = max(new_rows[i][-1], max(max(r) for r in new_rows if r is not new_rows[i]))
                # Optimization: new_cost is max of previous max and new_val
                # But we don't store previous max.
                # max(max(r) for r in new_rows) is O(n_rows * k)
                
                if new_cost < best_cost:
                    next_beam.append((new_cost, new_rows, new_used_diffs))
        
        if not next_beam:
            break
            
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]
        
        for cost, rows, _ in beam:
            if all(len(r) == row_len for r in rows):
                if cost < best_cost:
                    best_cost = cost
                    best_solution = [r[:] for r in rows]
                
    return best_solution
