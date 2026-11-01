import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
    min_m = n_rows * (k * (k + 1) // 2)
    
    def solve(m):
        used_diffs = set()
        rows = []
        
        def dfs(row_idx, pos, last_val, current_row, row_diffs_so_far):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == k + 1:
                # Row complete
                rows.append(current_row)
                
                # Add to global
                # We know row_diffs_so_far is the set of all diffs in this row
                if row_diffs_so_far & used_diffs:
                    rows.pop()
                    return False
                
                used_diffs.update(row_diffs_so_far)
                
                if dfs(row_idx + 1, 1, 0, [0], set()):
                    return True
                
                used_diffs.difference_update(row_diffs_so_far)
                rows.pop()
                return False
            
            max_v = m - (k - pos)
            if max_v < last_val + 1:
                return False
            
            for v in range(last_val + 1, max_v + 1):
                new_diffs = {v - a for a in current_row}
                
                # Check against global used
                if new_diffs & used_diffs:
                    continue
                
                # Check against current row's existing diffs
                if new_diffs & row_diffs_so_far:
                    continue
                
                # Valid
                current_row.append(v)
                # Update row_diffs_so_far
                # We need to add new_diffs to a copy of row_diffs_so_far
                # To avoid modifying the original set in the loop
                new_row_diffs = row_diffs_so_far.union(new_diffs)
                
                if dfs(row_idx, pos + 1, v, current_row, new_row_diffs):
                    return True
                
                current_row.pop()
            
            return False
        
        if dfs(0, 1, 0, [0], set()):
            return rows
        return None

    m = min_m
    while True:
        if time.time() > deadline:
            return greedy_construct(n_rows, k)
        
        sol = solve(m)
        if sol is not None:
            return sol
        m += 1

def greedy_construct(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
