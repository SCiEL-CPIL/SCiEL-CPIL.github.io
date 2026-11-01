def construct(n_rows, k):
    diffs_per_row = k * (k + 1) // 2
    
    def get_greedy():
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
        return rows, max(max(r) for r in rows)

    greedy_rows, M_greedy = get_greedy()
    
    if n_rows == 0 or k == 0:
        return greedy_rows

    M_lower = n_rows * diffs_per_row
    
    def is_feasible(M):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row, current_diffs):
            # Pruning: Check if enough differences remain
            req_current = diffs_per_row - len(current_diffs)
            req_future = (n_rows - row_idx - 1) * diffs_per_row
            total_req = req_current + req_future
            avail = M - len(used_diffs) - len(current_diffs)
            
            if avail < total_req:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                used_diffs.update(current_diffs)
                
                if backtrack(row_idx + 1, [0], set()):
                    return True
                
                used_diffs.difference_update(current_diffs)
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Try to find the next value
            # We can iterate x from last_val + 1 to M
            for x in range(last_val + 1, M + 1):
                new_diffs = {x - a for a in current_row}
                
                if new_diffs & used_diffs:
                    continue
                if new_diffs & current_diffs:
                    continue
                
                if backtrack(row_idx, current_row + [x], current_diffs.union(new_diffs)):
                    return True
            
            return False
        
        if backtrack(0, [0], set()):
            return rows
        return None

    # Binary search
    low = M_lower
    high = M_greedy
    best_rows = None
    
    # If the range is small, linear search might be faster or comparable
    # But binary search is standard.
    
    # To avoid long hangs on infeasible M near the upper bound,
    # we can also just try from M_lower upwards if the range is small.
    # M_greedy is usually not much larger than the optimum.
    
    # Let's try linear search from M_lower to M_greedy.
    # This guarantees we find the smallest M if we go up.
    # If M_greedy - M_lower is small (e.g., < 50), this is fine.
    # If it's large, binary search is better.
    
    if M_greedy - M_lower < 100:
        for M in range(M_lower, M_greedy + 1):
            res = is_feasible(M)
            if res:
                return res
    else:
        while low <= high:
            mid = (low + high) // 2
            res = is_feasible(mid)
            if res:
                best_rows = res
                high = mid - 1
            else:
                low = mid + 1
        if best_rows:
            return best_rows

    # Fallback
    return greedy_rows
