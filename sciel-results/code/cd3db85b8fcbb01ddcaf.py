import time
import random

def construct(n_rows, k):
    # Calculate lower bound for M
    total_diffs = n_rows * k * (k + 1) // 2
    M_min = total_diffs
    
    # Generate a greedy solution as a baseline/fallback
    def generate_greedy():
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

    greedy_rows = generate_greedy()
    greedy_max = max(max(r) for r in greedy_rows)
    
    # CSP Solver
    def find_solution_rows(M, time_limit):
        start_time = time.time()
        rows = []
        used_diffs = set()
        
        def backtrack(row_idx, current_row, row_diffs):
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row[:])
                for d in row_diffs:
                    used_diffs.add(d)
                
                if backtrack(row_idx + 1, [0], set()):
                    return True
                
                rows.pop()
                for d in row_diffs:
                    used_diffs.discard(d)
                return False
            
            last_val = current_row[-1]
            
            # Heuristic: Try to place elements that are "spread out"
            # Simple iteration is okay, but we can add a small randomization
            # to escape local minima in the search tree?
            # For now, deterministic.
            
            for next_val in range(last_val + 1, M + 1):
                valid = True
                new_diffs = []
                for prev_val in current_row:
                    d = next_val - prev_val
                    if d in row_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                for d in new_diffs:
                    if d in used_diffs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                current_row.append(next_val)
                old_row_diffs = set(row_diffs)
                row_diffs.update(new_diffs)
                
                if backtrack(row_idx, current_row, row_diffs):
                    return True
                
                current_row.pop()
                row_diffs = old_row_diffs
            
            return False
        
        if backtrack(0, [0], set()):
            return rows
        return None

    # Try to find a solution with M < greedy_max
    # We iterate M from M_min to greedy_max - 1
    # To avoid spending too much time on infeasible M, we limit time per M.
    # If greedy_max is close to M_min, we try them.
    
    # If greedy_max is very large, we might skip some.
    # But for this task, greedy_max is likely around 130-150.
    
    # We'll try up to 20 values of M to be safe.
    # If we find a solution, it's valid and better than greedy.
    
    # To increase chance of finding a solution, we can randomize the search order
    # in the CSP for each M.
    
    # Let's add randomization to the CSP candidate order.
    def find_solution_rows_randomized(M, time_limit):
        start_time = time.time()
        rows = []
        used_diffs = set()
        
        def backtrack(row_idx, current_row, row_diffs):
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row[:])
                for d in row_diffs:
                    used_diffs.add(d)
                
                if backtrack(row_idx + 1, [0], set()):
                    return True
                
                rows.pop()
                for d in row_diffs:
                    used_diffs.discard(d)
                return False
            
            last_val = current_row[-1]
            
            # Generate candidates
            candidates = list(range(last_val + 1, M + 1))
            # Randomize to explore different branches
            random.shuffle(candidates)
            
            for next_val in candidates:
                valid = True
                new_diffs = []
                for prev_val in current_row:
                    d = next_val - prev_val
                    if d in row_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                for d in new_diffs:
                    if d in used_diffs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                current_row.append(next_val)
                old_row_diffs = set(row_diffs)
                row_diffs.update(new_diffs)
                
                if backtrack(row_idx, current_row, row_diffs):
                    return True
                
                current_row.pop()
                row_diffs = old_row_diffs
            
            return False
        
        if backtrack(0, [0], set()):
            return rows
        return None

    # Try a few M values
    # We prioritize M values close to M_min.
    # We'll try M_min, M_min+1, ... up to greedy_max-1.
    # But we limit total time to ~100s to leave buffer.
    
    global_start = time.time()
    max_search_time = 100.0
    
    for M in range(M_min, greedy_max):
        if time.time() - global_start > max_search_time:
            break
            
        # Limit time per M to 2s to ensure we try multiple M
        sol = find_solution_rows_randomized(M, time_limit=2.0)
        if sol:
            return sol
            
    return greedy_rows
