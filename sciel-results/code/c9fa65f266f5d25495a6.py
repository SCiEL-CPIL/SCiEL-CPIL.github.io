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
