import random
import math
import time
import heapq

def construct(n_rows, k):
    row_len = k + 1
    
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def is_valid_extension(row, new_val, used_diffs):
        if new_val <= row[-1]:
            return False
        new_diffs = {new_val - v for v in row}
        if new_diffs & used_diffs:
            return False
        return True

    def greedy_solution():
        rows = [[0] for _ in range(n_rows)]
        used = set()
        for step in range(n_rows * k):
            # Pick row with smallest length, then smallest last val
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
                break
            i = target_row_idx
            current_row = rows[i]
            x = current_row[-1] + 1
            while True:
                if is_valid_extension(current_row, x, used):
                    current_row.append(x)
                    for v in current_row[:-1]:
                        used.add(x - v)
                    break
                x += 1
        return rows, max(max(r) for r in rows)

    def gap_penalty(used_diffs, max_val):
        # Count how many integers in [1, max_val] are NOT in used_diffs
        # This is a measure of "sparseness". We want to minimize this (maximize density).
        # But calculating this exactly is O(max_val).
        # Approximation: 
        # If we assume used_diffs is a set of size S, and max_val is M.
        # Density = S / M.
        # Penalty = M - S.
        # We want to minimize M - S.
        # Since S is fixed by the number of steps taken (in beam search, all states at same depth have same S),
        # minimizing M - S is equivalent to minimizing M.
        # So this penalty is redundant if we already minimize MaxEntry.
        # However, different paths might have different MaxEntry and different "potential" for future small diffs.
        # Let's use a different metric: 
        # "Wasted Space": The number of small integers that are skipped.
        # This is hard to compute quickly.
        
        # Alternative: Just use MaxEntry.
        return 0

    start_time = time.time()
    timeout = 110
    
    baseline_rows, baseline_cost = greedy_solution()
    best_solution = baseline_rows
    best_cost = baseline_cost
    
    beam_width = 3000 
    max_depth = n_rows * k
    
    initial_rows = [[0] for _ in range(n_rows)]
    initial_used = set()
    
    # State: (cost, rows, used_diffs)
    # We sort by cost.
    beam = [(0, initial_rows, initial_used)]
    
    for step in range(n_rows * k):
        if time.time() - start_time > timeout:
            break
            
        next_beam = []
        
        for cost, rows, used_diffs in beam:
            # Identify the row to expand
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
            
            # Find valid candidates
            limit_val = best_cost - 1
            
            candidates = []
            x = current_row[-1] + 1
            while x <= limit_val:
                if is_valid_extension(current_row, x, used_diffs):
                    candidates.append(x)
                    if len(candidates) >= 30: # Increased candidates to allow better choice
                        break
                x += 1
                
            if not candidates:
                continue

            # Heuristic: 
            # We prefer candidates that result in a lower new_max.
            # If new_max is the same, we prefer candidates that create "smaller" new differences?
            # Actually, creating smaller differences is good for density.
            
            scored_candidates = []
            for new_val in candidates:
                new_rows = [r[:] for r in rows]
                new_rows[i].append(new_val)
                
                new_used_diffs = set(used_diffs)
                new_diffs_added = []
                for v in new_rows[i]:
                    if v != new_val:
                        d = abs(new_val - v)
                        new_used_diffs.add(d)
                        new_diffs_added.append(d)
                        
                new_cost = max(max(r) for r in new_rows)
                
                # Secondary metric: Sum of new differences?
                # Smaller sum of new differences means we are using "cheaper" differences.
                # This helps keep the global difference set compact.
                sec_cost = sum(new_diffs_added)
                
                if new_cost < best_cost:
                    # We store (primary_cost, secondary_cost, state)
                    next_beam.append((new_cost, sec_cost, new_rows, new_used_diffs))
        
        if not next_beam:
            break
            
        # Sort by primary cost, then secondary cost
        next_beam.sort(key=lambda x: (x[0], x[1]))
        
        # Prune to beam width
        beam = []
        seen_states = set() # Optional: avoid duplicate states if possible, but rows are unique enough
        for item in next_beam:
            if len(beam) >= beam_width:
                break
            # Add to beam
            beam.append(item)
            
        # Check for complete solutions in the new beam
        for cost, sec_cost, rows, _ in beam:
            if all(len(r) == row_len for r in rows):
                if cost < best_cost:
                    best_cost = cost
                    best_solution = [r[:] for r in rows]
                
    if best_solution is None:
        # Fallback
        rows = [[0] for _ in range(n_rows)]
        used = set()
        for _ in range(n_rows * k):
            i = min(range(n_rows), key=lambda idx: (len(rows[idx]), rows[idx][-1]))
            if len(rows[i]) >= row_len:
                continue
            x = rows[i][-1] + 1
            while True:
                new_diffs = {x - v for v in rows[i]}
                if not (new_diffs & used):
                    rows[i].append(x)
                    used.update(new_diffs)
                    break
                x += 1
        best_solution = rows
        best_cost = max(max(r) for r in rows)

    return best_solution
