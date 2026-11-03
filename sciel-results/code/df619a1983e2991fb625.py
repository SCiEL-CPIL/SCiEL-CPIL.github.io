import time

def construct(n_rows, k):
    # Baseline: Greedy
    def greedy():
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

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    start_time = time.time()
    timeout = 115.0
    
    # Precompute binomials
    binom = [[0]*(k+2) for _ in range(k+2)]
    for n in range(k+2):
        for r in range(n+1):
            if r == 0 or r == n:
                binom[n][r] = 1
            else:
                binom[n][r] = binom[n-1][r-1] + binom[n-1][r]
    
    diffs_per_row = binom[k+1][2]
    
    # Helper to check if we can still fit the required differences
    # within the limit using bitmask
    def can_complete(i, row, used_mask, limit):
        cur_len = len(row)
        diffs_cur_rem = diffs_per_row - binom[cur_len][2]
        rows_rem = n_rows - i - 1
        diffs_full_rem = rows_rem * diffs_per_row
        total_need = diffs_cur_rem + diffs_full_rem
        
        if total_need == 0:
            return True
            
        avail = 0
        # Iterate d from 1 to limit
        # Optimization: check bits
        for d in range(1, limit + 1):
            if not (used_mask & (1 << d)):
                avail += 1
                if avail >= total_need:
                    return True
        return avail >= total_need

    # DFS with Pruning
    def dfs(i, row, used_mask, limit):
        if time.time() - start_time > timeout:
            return "timeout", None
            
        if row and row[-1] > limit:
            return False, None
            
        if not can_complete(i, row, used_mask, limit):
            return False, None
            
        if i == n_rows:
            return True, []
            
        if len(row) < k + 1:
            last = row[-1]
            for x in range(last + 1, limit + 1):
                new_mask = 0
                valid = True
                for a in row:
                    d = x - a
                    if (used_mask & (1 << d)):
                        valid = False
                        break
                    new_mask |= (1 << d)
                
                if valid:
                    new_row = row + [x]
                    new_used = used_mask | new_mask
                    status, sub_rows = dfs(i, new_row, new_used, limit)
                    if status == True:
                        return True, [new_row] + sub_rows
                    elif status == "timeout":
                        return "timeout", None
            
            return False, None
        else:
            new_i = i + 1
            new_row = [0]
            status, sub_rows = dfs(new_i, new_row, used_mask, limit)
            if status == True:
                return True, [row] + sub_rows
            elif status == "timeout":
                return "timeout", None
            else:
                return False, None

    current_best_rows = best_rows
    current_best_scope = best_scope
    
    # Try to improve by searching for smaller scopes
    # Start from best_scope - 1
    limit = best_scope - 1
    
    while limit > 0:
        if time.time() - start_time > timeout:
            break
            
        status, rows = dfs(0, [0], 0, limit)
        
        if status == "timeout":
            break
        elif status == True:
            actual_scope = max(max(r) for r in rows)
            current_best_rows = rows
            current_best_scope = actual_scope
            # Try to improve further
            limit = actual_scope - 1
        else:
            # No solution found for this limit, so no solution for lower limits
            break
            
    return current_best_rows
