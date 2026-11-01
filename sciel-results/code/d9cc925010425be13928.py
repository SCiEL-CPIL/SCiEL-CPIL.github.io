import sys

def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def construct(n_rows, k):
    L = k + 1
    diffs_per_row = L * (L - 1) // 2
    total_diffs = n_rows * diffs_per_row
    min_M = total_diffs
    
    # Search for optimal M
    # Upper bound: min_M + 50 is safe for small instances
    # For L=6, n=7, min_M=105. Optimal is 112.
    
    sys.setrecursionlimit(10000)
    
    global solution_found
    global final_rows
    
    for M in range(min_M, min_M + 100):
        solution_found = False
        final_rows = None
        used = set()
        rows = [[0] for _ in range(n_rows)]
        
        def dfs(r, c, last_val):
            if solution_found:
                return True
                
            if r == n_rows:
                solution_found = True
                final_rows = [row[:] for row in rows]
                return True
                
            if c == L:
                # Row complete
                if r > 0:
                    if tuple(rows[r]) < tuple(rows[r-1]):
                        return False
                return dfs(r + 1, 1, 0)
            
            min_next = last_val + 1
            
            if r > 0:
                prev = rows[r-1]
                curr = rows[r]
                # Check lexicographic constraint
                # curr[0:c] vs prev[0:c]
                is_greater = False
                is_smaller = False
                for i in range(c):
                    if curr[i] > prev[i]:
                        is_greater = True
                        break
                    elif curr[i] < prev[i]:
                        is_smaller = True
                        break
                
                if is_smaller:
                    return False
                elif not is_greater:
                    # Prefixes equal
                    if c < L:
                        min_next = max(min_next, prev[c])
            
            # Max next val to allow remaining slots
            remaining = L - c - 1
            max_next = M - remaining
            
            if min_next > max_next:
                return False
                
            for v in range(min_next, max_next + 1):
                # Get current values in row r
                # rows[r] has length c
                current_vals = rows[r]
                new_diffs = {v - a for a in current_vals}
                
                if new_diffs & used:
                    continue
                
                # Place
                rows[r].append(v)
                used.update(new_diffs)
                
                if dfs(r, c + 1, v):
                    return True
                
                # Backtrack
                rows[r].pop()
                used.difference_update(new_diffs)
                
            return False
        
        if dfs(0, 1, 0):
            return final_rows
            
    return greedy_construct(n_rows, k)
