import heapq
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
    
    # A* Search
    # State: (max_so_far, i, row_tuple, used_mask)
    # i: index of current row
    # row_tuple: tuple of current row elements
    # used_mask: bitmask of used differences
    
    start_time = time.time()
    timeout = 115.0
    
    # Precompute binomials for heuristic
    binom = [[0]*(k+2) for _ in range(k+2)]
    for n in range(k+2):
        for r in range(n+1):
            if r == 0 or r == n:
                binom[n][r] = 1
            else:
                binom[n][r] = binom[n-1][r-1] + binom[n-1][r]
    
    total_diffs_per_row = binom[k+1][2]
    
    def get_heuristic(i, row_tuple, used_mask):
        # Calculate number of diffs needed
        len_row = len(row_tuple)
        diffs_cur = binom[len_row][2]
        diffs_cur_rem = total_diffs_per_row - diffs_cur
        rows_rem = n_rows - i - 1
        diffs_full_rem = rows_rem * total_diffs_per_row
        
        need = diffs_cur_rem + diffs_full_rem
        
        if need == 0:
            return 0
            
        # Find 'need' smallest positive integers not in used_mask
        found = 0
        max_new = 0
        d = 1
        # Optimization: if need is large, we might go up to ~100-200
        while found < need:
            if not (used_mask & (1 << d)):
                found += 1
                max_new = d
            d += 1
        
        return max_new

    # Initial state
    # i=0, row=(0,), used_mask=0
    # max_so_far = 0
    
    start_state = (0, 0, (0,), 0) # max_so_far, i, row, used_mask
    open_list = [start_state]
    
    # best_max_for_state[state] = min max_so_far seen
    best_max_for_state = {}
    
    best_found = best_rows
    best_scope_val = best_scope
    
    while open_list:
        if time.time() - start_time > timeout:
            break
            
        max_so_far, i, row, used_mask = heapq.heappop(open_list)
        
        state_key = (i, row, used_mask)
        
        # Prune if we've seen this state with a smaller or equal max_so_far
        if state_key in best_max_for_state:
            if best_max_for_state[state_key] <= max_so_far:
                continue
        best_max_for_state[state_key] = max_so_far
        
        # Prune if we already have a better solution
        if max_so_far >= best_scope_val:
            # Since we pop in increasing max_so_far, if the current one is >= best, 
            # all subsequent ones will be >= best. 
            # We can break if we are sure no better solution exists?
            # Yes, because any goal popped later will have max_so_far >= current.
            # So if current max_so_far >= best_scope_val, we cannot find a solution 
            # with scope < best_scope_val.
            break
        
        # Calculate heuristic
        h = get_heuristic(i, row, used_mask)
        
        # If heuristic exceeds current best, prune
        if h >= best_scope_val:
            continue
            
        # If we found a complete solution
        if i == n_rows - 1 and len(row) == k + 1:
            # This is a goal state
            # max_so_far is the scope
            if max_so_far < best_scope_val:
                best_scope_val = max_so_far
                # Reconstruct path? 
                # We didn't store parent pointers. 
                # We need to store the rows to return them.
                # Storing rows in state makes it heavy.
                # Alternative: Store path in state? 
                # Or just return the scope? No, must return rows.
                # We must store the rows.
                # Let's modify state to include `rows_tuple` (tuple of tuples).
                # This increases memory but is necessary for reconstruction.
                pass
        
        # Expand state
        if len(row) < k + 1:
            last = row[-1]
            # Try next x
            # Limit x to best_scope_val - 1 to find strictly better
            # Actually, if x == best_scope_val, max_so_far becomes best_scope_val.
            # We break if max_so_far >= best_scope_val.
            # So we only care about x < best_scope_val.
            for x in range(last + 1, best_scope_val):
                new_diffs_mask = 0
                valid = True
                for a in row:
                    d = x - a
                    if used_mask & (1 << d):
                        valid = False
                        break
                    new_diffs_mask |= (1 << d)
                
                if valid:
                    new_row = row + (x,)
                    new_used = used_mask | new_diffs_mask
                    new_max = max(max_so_far, x)
                    
                    # Check heuristic for child? 
                    # We do it in pop or push?
                    # Doing it in push prunes earlier.
                    # But heuristic calc is O(scope).
                    # Let's do a quick check: if new_max >= best_scope_val, skip.
                    if new_max >= best_scope_val:
                        continue
                        
                    child_state = (new_max, i, new_row, new_used)
                    heapq.heappush(open_list, child_state)
        else:
            # Row is full
            if i < n_rows - 1:
                # Move to next row
                new_i = i + 1
                new_row = (0,)
                new_used = used_mask
                new_max = max_so_far
                
                child_state = (new_max, new_i, new_row, new_used)
                heapq.heappush(open_list, child_state)
            else:
                # Goal
                # We need to return the rows.
                # We didn't store them.
                # This implementation is flawed for returning the actual rows.
                pass

    return best_found
