import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Fallback greedy function
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

    # If n_rows or k is large, backtracking might be too slow.
    # For this task (7, 6), it's fine.
    # But to be safe, if k > 8 or n_rows > 10, just use greedy.
    if k > 10 or n_rows > 10:
        return greedy_construct(n_rows, k)

    bound = 120
    
    # Generate all valid rows (Sidon sets of size k+1 starting at 0) 
    # with max element <= bound.
    
    def gen_sidon_bitset(k, max_val):
        results = []
        
        def dfs(idx, last_val, current_row, diff_mask):
            if time.time() > deadline:
                return
            
            if idx == k + 1:
                max_elem = current_row[-1]
                results.append((tuple(current_row), max_elem, diff_mask))
                return
            
            # Pruning: 
            remaining = k + 1 - idx
            if last_val + remaining > max_val:
                return
            
            for x in range(last_val + 1, max_val + 1):
                new_mask = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if new_mask & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if conflict:
                    continue
                
                if diff_mask & new_mask:
                    continue
                
                dfs(idx + 1, x, current_row + [x], diff_mask | new_mask)
        
        dfs(1, 0, [0], 0)
        return results

    try:
        all_rows = gen_sidon_bitset(k, bound)
    except Exception:
        all_rows = []

    if not all_rows:
        return greedy_construct(n_rows, k)

    # Sort rows by max element
    all_rows.sort(key=lambda x: x[1])
    
    best_solution = None
    best_max = float('inf')
    
    def search(selected_rows, used_mask, start_idx):
        nonlocal best_solution, best_max
        
        if time.time() > deadline:
            return
        
        if len(selected_rows) == n_rows:
            current_max = max(r[1] for r in selected_rows)
            if current_max < best_max:
                best_max = current_max
                best_solution = [list(r[0]) for r in selected_rows]
            return
        
        # Pruning: if the smallest max element available from start_idx is >= best_max, stop
        if all_rows[start_idx][1] >= best_max:
            return
        
        for i in range(start_idx, len(all_rows)):
            row, max_elem, mask = all_rows[i]
            
            if max_elem >= best_max:
                break
            
            if used_mask & mask:
                continue
            
            search(selected_rows + [row], used_mask | mask, i + 1)
            
            # Optional: if we found a solution, we can break early if we want just one?
            # No, we want the best (min max). 
            # But since we sorted by max, the first complete solution found 
            # in a DFS that explores in order of max might not be the global min 
            # if the DFS doesn't explore all branches with smaller max first?
            # Actually, the loop `for i in range(start_idx, len(all_rows))` 
            # tries rows in increasing order of max.
            # So it will try to build a solution using the smallest max rows first.
            # If it finds a solution, that solution's max is determined by the largest row in it.
            # If we find a solution with max M, we set best_max = M.
            # Then we prune any branch that involves a row with max >= M.
            # This ensures we only look for solutions with max < M.
            # So the first solution found might not be the best, but the pruning 
            # will guide the search to find better ones if they exist.
            # Wait, if we find a solution with max 112, we prune >= 112.
            # We continue searching for solutions with max < 112.
            # If we find one with 110, we update best_max to 110 and prune >= 110.
            # This is correct.

    search([], 0, 0)
    
    if best_solution:
        return best_solution
    else:
        return greedy_construct(n_rows, k)

# EVOLVE-BLOCK-END
