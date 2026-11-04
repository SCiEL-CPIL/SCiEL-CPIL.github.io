import time

def construct(n_rows, k):
    start_time = time.time()
    # Leave 5 seconds for cleanup/return
    deadline = start_time + 115 

    # 1. Greedy Fallback
    def greedy_rulers(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_rulers(n_rows, k)
    best_M = max(max(row) for row in best_rows)
    
    # 2. Search for Optimal
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    M_min = max(diffs_per_row, total_diffs)
    
    # Search M from M_min to best_M - 1
    # We use a shared deadline
    search_deadline = start_time + 110 # 10s margin for overhead
    
    for M in range(M_min, best_M):
        if time.time() > search_deadline:
            break
        
        found = search_solution(n_rows, k, M, search_deadline)
        if found is not None:
            best_rows = found
            best_M = M
            break
            
    return best_rows

def search_solution(n_rows, k, M, deadline):
    found_rows = [None]
    
    # Precompute binom
    def nCr(n, r):
        if r < 0 or r > n: return 0
        if r == 0 or r == n: return 1
        return math.comb(n, r)
        
    import math
    binom_k1_2 = math.comb(k+1, 2)
    
    def dfs(rows_remaining, current_row, used_diffs, rows_so_far):
        if time.time() > deadline:
            return False
            
        if rows_remaining == 0:
            found_rows[0] = rows_so_far
            return True
            
        L = len(current_row)
        
        if L == k + 1:
            # Move to next row
            return dfs(rows_remaining - 1, [0], used_diffs, rows_so_far + [current_row])
            
        # Pruning: Check if enough differences remain
        # Diffs needed for current row completion
        diffs_needed_current = binom_k1_2 - math.comb(L, 2)
        # Diffs needed for remaining full rows
        diffs_needed_future = (rows_remaining - 1) * binom_k1_2
        total_diffs_needed = diffs_needed_current + diffs_needed_future
        
        available_diffs = M - len(used_diffs)
        if total_diffs_needed > available_diffs:
            return False
            
        last_val = current_row[-1]
        
        # Iterate candidates
        # Optimization: We can restrict x.
        # The smallest possible next difference is 1.
        # The largest difference we can add is M - 0 = M.
        # x must be > last_val.
        
        for x in range(last_val + 1, M + 1):
            new_diffs = {x - a for a in current_row}
            
            # Check if any new diff is already used
            # Using set intersection is fast
            if new_diffs & used_diffs:
                continue
                
            # Recurse
            # Optimization: Instead of creating new set, we can add/remove
            # But for simplicity and safety with recursion depth, creating new set is okay for small M.
            # However, to speed up, we can use a list-based boolean array and backtrack.
            # Given M < 200, set union is fine.
            
            if dfs(rows_remaining, current_row + [x], used_diffs | new_diffs, rows_so_far):
                return True
                
        return False
        
    dfs(n_rows, [0], set(), [])
    return found_rows[0]
