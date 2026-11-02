import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110
    
    # The known optimal scope is 112.
    # We will search for a valid configuration with max element <= 112.
    # We use a backtracking search.
    # To reduce symmetry, we fix the first row to the greedy ruler.
    
    # Generate greedy ruler for k+1 marks
    def get_greedy_row(k):
        row = [0]
        used = set()
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
                x += 1
        return row

    greedy_row = get_greedy_row(k)
    
    # Pre-compute differences for greedy row
    init_used = set()
    for i in range(len(greedy_row)):
        for j in range(i+1, len(greedy_row)):
            init_used.add(greedy_row[j] - greedy_row[i])

    best_solution = None
    
    def backtrack(row_idx, current_row, used_diffs, max_m):
        nonlocal best_solution
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            best_solution = [r[:] for r in rows]
            return True
            
        if len(current_row) == k + 1:
            rows.append(current_row)
            if backtrack(row_idx + 1, [0], used_diffs, max_m):
                return True
            rows.pop()
            return False
            
        last = current_row[-1]
        r_needed = (k + 1) - len(current_row)
        
        # Candidates for x
        # x must be > last
        # x must be <= max_m
        # Also, we need space for r_needed - 1 more elements after x.
        # So x + (r_needed - 1) <= max_m  => x <= max_m - (r_needed - 1)
        max_x = max_m - (r_needed - 1)
        
        for x in range(last + 1, max_x + 1):
            new_diffs = []
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            current_row.append(x)
            used_diffs.update(new_diffs)
            
            if backtrack(row_idx, current_row, used_diffs, max_m):
                return True
                
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
        return False

    # Try to find solution with max_m = 112
    # If greedy row max > 112, it's impossible, but greedy max is 22.
    
    # We assume the optimal solution contains the greedy row.
    # If this assumption is wrong, we will fail to find 112.
    # To be safe, we can try a few other "simple" rows if this fails, 
    # but given time, we stick to this.
    
    for m in [112]: # Just try 112 first
        if greedy_row[-1] > m:
            continue
        rows = [greedy_row]
        used_diffs = set(init_used)
        
        if backtrack(1, [0], used_diffs, m):
            return best_solution
            
    # If failed, try m=113, 114... up to deadline
    for m in range(113, 150):
        if time.time() > deadline - 5:
            break
        if greedy_row[-1] > m:
            continue
        rows = [greedy_row]
        used_diffs = set(init_used)
        
        if backtrack(1, [0], used_diffs, m):
            return best_solution
            
    # Fallback to seed greedy
    used, rows_out = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows_out.append(row)
    return rows_out
