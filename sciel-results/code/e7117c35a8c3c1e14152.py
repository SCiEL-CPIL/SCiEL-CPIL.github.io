<<<<<<< SEARCH
import heapq
import itertools
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
    # State: (f, h, i, row, used_mask, path)
    # i: index of current row
    # row: current partial row
    # used_mask: bitmask of used differences
    # path: list of completed rows
    
    start_time = time.time()
    timeout = 115.0 # Leave 5s buffer
    
    # Precompute binomials
    binom = [[0]*(k+2) for _ in range(k+2)]
    for n in range(k+2):
        for r in range(n+1):
            if r == 0 or r == n:
                binom[n][r] = 1
            else:
                binom[n][r] = binom[n-1][r-1] + binom[n-1][r]
    
    def get_heuristic(i, row, used_mask):
        # Calculate number of diffs needed
        # Diffs in current row so far: binom[len(row)][2]
        # Diffs needed to finish current row: binom[k+1][2] - binom[len(row)][2]
        diffs_cur_rem = binom[k+1][2] - binom[len(row)][2]
        # Diffs needed for remaining full rows
        rows_rem = n_rows - i - 1
        diffs_full_rem = rows_rem * binom[k+1][2]
        
        need = diffs_cur_rem + diffs_full_rem
        
        if need == 0:
            return 0
            
        # Find 'need' smallest positive integers not in used_mask
        # used_mask has bits set for used diffs.
        # We iterate d from 1 upwards.
        found = 0
        max_new = 0
        d = 1
        while found < need:
            if not (used_mask & (1 << d)):
                found += 1
                max_new = d
            d += 1
        
        # Max used diff
        max_used = 0
        if used_mask:
            max_used = used_mask.bit_length()
            
        return max(max_used, max_new)

    # Initial state
    # i=0, row=[0], used_mask=0, path=[]
    # Note: The difference set for [0] is empty.
    start_state = (0, 0, 0, [0], 0, []) # f, h, i, row, used_mask, path
    # f = g + h. g=0. h = get_heuristic(0, [0], 0)
    start_h = get_heuristic(0, [0], 0)
    start_state = (start_h, start_h, 0, [0], 0, [])
    
    open_list = [start_state]
    visited = set() # (i, tuple(row), used_mask)
    
    best_found = best_rows
    best_scope_val = best_scope
    
    while open_list:
        if time.time() - start_time > timeout:
            break
            
        f, h, i, row, used_mask, path = heapq.heappop(open_list)
        
        state_key = (i, tuple(row), used_mask)
        if state_key in visited:
            continue
        visited.add(state_key)
        
        # If we found a complete solution
        if i == n_rows:
            # path contains all rows
            scope = max(max(r) for r in path)
            if scope < best_scope_val:
                best_scope_val = scope
                best_found = path
                # Since A* pops in order of f, and h is admissible,
                # the first goal popped is optimal?
                # Yes, if h is admissible and consistent.
                # Our h is admissible. Is it consistent?
                # Likely yes, as adding a diff increases the lower bound on scope monotonically.
                # So we can return immediately?
                # But we want to be safe and maybe find a slightly better one if h was not perfect?
                # No, A* guarantees optimality if h is admissible.
                # So the first goal is the optimal one.
                return path
        
        # Expand state
        # Case 1: Current row is not full
        if len(row) < k + 1:
            last = row[-1]
            # Try next x
            # We can limit x to best_scope_val to prune
            # But we don't know the exact limit for this branch.
            # Let's try x from last+1 to best_scope_val
            for x in range(last + 1, best_scope_val + 1):
                new_diffs = [x - a for a in row]
                # Check if any diff is used
                valid = True
                new_mask = 0
                for d in new_diffs:
                    if used_mask & (1 << d):
                        valid = False
                        break
                    new_mask |= (1 << d)
                
                if valid:
                    new_row = row + [x]
                    new_used = used_mask | new_mask
                    new_i = i
                    new_path = path
                    
                    new_h = get_heuristic(new_i, new_row, new_used)
                    new_g = f - h # f = g + h, so g = f - h. 
                    # Wait, f is the cost to reach this node + heuristic?
                    # In standard A*, f = g + h.
                    # We stored f and h. So g = f - h.
                    # For child: g_child = g + 1? No, cost is not 1.
                    # The "cost" in A* is usually the path length.
                    # Here, we want to minimize the scope.
                    # This is not a standard shortest path with uniform edge costs.
                    # The "cost" is the max value.
                    # This is a minimax problem.
                    # A* with minimax objective:
                    # f(state) = max(g(state), h(state))?
                    # Or we can define the cost of the path as the max element in the path so far.
                    # Let g(state) = max element in `path` + `row`.
                    # Then f(state) = max(g(state), h(state)).
                    
                    # Let's re-evaluate.
                    # g = max(max(r) for r in path) if path else 0
                    # g = max(g, max(row))
                    
                    # We need to store g in the state or calculate it.
                    # Let's add g to the state tuple.
                    pass
        
        # Case 2: Current row is full
        if len(row) == k + 1 and i < n_rows - 1:
            # Move to next row
            new_i = i + 1
            new_row = [0]
            new_used = used_mask
            new_path = path + [row]
            
            # g is max of path
            g = max(max(r) for r in new_path) if new_path else 0
            
            new_h = get_heuristic(new_i, new_row, new_used)
            new_f = max(g, new_h)
            
            child_state = (new_f, new_h, new_i, new_row, new_used, new_path)
            heapq.heappush(open_list, child_state)

    return best_found
=======
import heapq
import time

def construct(n_rows, k):
    # Baseline: Greedy to get an initial upper bound
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
    timeout = 115.0  # Leave 5s buffer
    
    # Precompute binomial coefficients
    def binom(n, r):
        if r < 0 or r > n:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        res = 1
        for i in range(r):
            res = res * (n - i) // (i + 1)
        return res
    
    diffs_per_row = binom(k + 1, 2)
    
    # Heuristic: given used_mask and number of additional diffs needed,
    # find the minimum possible maximum difference value
    def heuristic(used_mask, needed_diffs):
        if needed_diffs == 0:
            # The scope is at least the max used diff
            return used_mask.bit_length() if used_mask else 0
        
        # Find the smallest M such that there are at least needed_diffs
        # differences in [1, M] not in used_mask
        # Binary search on M
        lo = used_mask.bit_length() if used_mask else 1
        hi = lo + needed_diffs + 10  # Upper bound: worst case all used
        
        while lo < hi:
            mid = (lo + hi) // 2
            # Count available diffs in [1, mid]
            available = mid
            if used_mask:
                # Count bits set in used_mask that are <= mid
                mask_truncated = used_mask & ((1 << (mid + 1)) - 1)
                used_count = mask_truncated.bit_count()
                available -= used_count
            
            if available >= needed_diffs:
                hi = mid
            else:
                lo = mid + 1
        
        return lo
    
    # A* search
    # State: (f, i, row, used_mask, g_scope)
    # f = max(g_scope, h)
    # i: current row index
    # row: current partial row (list)
    # used_mask: bitmask of used differences
    # g_scope: max element in any completed row + current row so far
    
    # Initial state: row 0, partial row [0], no diffs used, scope 0
    initial_h = heuristic(0, n_rows * diffs_per_row)
    open_list = [(initial_h, 0, [0], 0, 0)]  # (f, i, row, used_mask, g_scope)
    visited = set()
    
    while open_list:
        if time.time() - start_time > timeout:
            break
        
        f, i, row, used_mask, g_scope = heapq.heappop(open_list)
        
        state_key = (i, tuple(row), used_mask)
        if state_key in visited:
            continue
        visited.add(state_key)
        
        # Check if we've completed all rows
        if i == n_rows:
            # All rows are complete, g_scope is the final scope
            if g_scope < best_scope:
                best_scope = g_scope
                # Reconstruct solution from path - but we don't store path
                # So we need to store the rows. Let's modify to store path.
                pass
        
        # Expand state
        # Case 1: Current row is not full, add next element
        if len(row) < k + 1:
            last = row[-1]
            # Try adding x from last+1 up to best_scope
            for x in range(last + 1, best_scope + 1):
                new_diffs = [x - a for a in row]
                # Check if any diff conflicts
                valid = True
                new_mask_bits = 0
                for d in new_diffs:
                    if used_mask & (1 << d):
                        valid = False
                        break
                    new_mask_bits |= (1 << d)
                
                if valid:
                    new_row = row + [x]
                    new_used = used_mask | new_mask_bits
                    new_g_scope = max(g_scope, x)
                    
                    # Calculate heuristic for remaining diffs
                    # Diffs used so far in this row: binom(len(new_row), 2)
                    # Diffs needed to complete this row: diffs_per_row - binom(len(new_row), 2)
                    # Diffs needed for remaining rows: (n_rows - i - 1) * diffs_per_row
                    cur_row_diffs = binom(len(new_row), 2)
                    cur_row_rem = diffs_per_row - cur_row_diffs
                    rem_rows = n_rows - i - 1
                    total_needed = cur_row_rem + rem_rows * diffs_per_row
                    
                    new_h = heuristic(new_used, total_needed)
                    new_f = max(new_g_scope, new_h)
                    
                    # Prune if not better than current best
                    if new_f < best_scope:
                        heapq.heappush(open_list, (new_f, i, new_row, new_used, new_g_scope))
        
        # Case 2: Current row is full, move to next row
        if len(row) == k + 1 and i < n_rows - 1:
            new_i = i + 1
            new_row = [0]
            new_used = used_mask
            new_g_scope = g_scope  # scope doesn't change when starting new row
            
            # Diffs needed for remaining rows
            rem_rows = n_rows - new_i - 1
            total_needed = rem_rows * diffs_per_row
            
            new_h = heuristic(new_used, total_needed)
            new_f = max(new_g_scope, new_h)
            
            if new_f < best_scope:
                heapq.heappush(open_list, (new_f, new_i, new_row, new_used, new_g_scope))
    
    return best_rows
>>>>>>> REPLACE

Wait, I realize the above doesn't actually reconstruct the solution properly - it just returns the greedy solution. Let me fix this to properly track the path and return the improved solution when found.

