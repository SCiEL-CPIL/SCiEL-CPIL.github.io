import time
import random
import sys

sys.setrecursionlimit(10000)

def construct_greedy(n_rows, k):
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

def construct(n_rows, k):
    # Initialize with greedy solution
    best_rows = construct_greedy(n_rows, k)
    best_max_val = max(max(row) for row in best_rows)
    
    start_time = time.time()
    time_limit = 110.0 # Leave 40s margin for safety
    
    search_best_rows = best_rows
    search_best_max = best_max_val
    
    def backtrack(row_idx, mark_idx, current_row, used_diffs, current_max, completed_rows):
        nonlocal search_best_rows, search_best_max
        
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            if current_max < search_best_max:
                search_best_max = current_max
                search_best_rows = completed_rows + [current_row[:]]
            return True
        
        if mark_idx == k + 1:
            return backtrack(row_idx + 1, 0, [0], used_diffs, current_max, completed_rows + [current_row[:]])
        
        last_mark = current_row[-1]
        
        # Find valid candidates
        candidates = []
        x = last_mark + 1
        # We can limit the search range to search_best_max
        while x <= search_best_max:
            new_diffs = {x - a for a in current_row}
            if not (new_diffs & used_diffs):
                candidates.append(x)
                if len(candidates) >= 10: # Cap candidates to keep it fast
                    break
            x += 1
            
        if not candidates:
            return False
            
        # Strategy: Mostly pick the smallest, sometimes a random one
        # This helps in escaping local minima where the smallest choice leads to a dead end
        if random.random() < 0.8:
            chosen_x = candidates[0]
        else:
            chosen_x = random.choice(candidates)
            
        # Try the chosen candidate
        new_diffs = {chosen_x - a for a in current_row}
        new_used = used_diffs | new_diffs
        new_row = current_row + [chosen_x]
        new_max = max(current_max, chosen_x)
        
        if backtrack(row_idx, mark_idx + 1, new_row, new_used, new_max, completed_rows):
            return True
            
        # If the chosen candidate failed, try the others
        for x in candidates:
            if x == chosen_x:
                continue
            new_diffs = {x - a for a in current_row}
            new_used = used_diffs | new_diffs
            new_row = current_row + [x]
            new_max = max(current_max, x)
            
            if backtrack(row_idx, mark_idx + 1, new_row, new_used, new_max, completed_rows):
                return True
                
        return False

    try:
        backtrack(0, 0, [0], set(), 0, [])
    except Exception:
        pass # Return greedy if backtracking fails

    return search_best_rows
