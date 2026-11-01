import sys
import time

def construct(n_rows, k):
    # Calculate lower bound for M
    total_diffs_needed = n_rows * (k + 1) * k // 2
    M_min = 1
    while M_min * (M_min + 1) // 2 < total_diffs_needed:
        M_min += 1
        
    # Greedy fallback
    def greedy_construct(n_rows, k):
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

    greedy_rows = greedy_construct(n_rows, k)
    M_max = max(max(row) for row in greedy_rows)
    
    # Search for minimal M
    # To avoid timeout, we limit the search time or nodes
    start_time = time.time()
    node_limit = 100000 # Adjust based on speed
    
    for M in range(M_min, M_max + 1):
        if time.time() - start_time > 110: # Leave 10s buffer
            break
        result = backtrack_solve(n_rows, k, M, node_limit)
        if result is not None:
            return result
            
    return greedy_rows

def backtrack_solve(n_rows, k, M, node_limit):
    used_diffs = set()
    rows = []
    nodes_visited = 0
    limit_exceeded = [False]
    
    total_diffs_per_row = (k + 1) * k // 2
    total_diffs_all = n_rows * total_diffs_per_row
    max_possible_diffs = M * (M + 1) // 2
    
    def solve_rows(row_idx, current_row, last_val):
        nonlocal nodes_visited, limit_exceeded
        if limit_exceeded[0]:
            return False
        nodes_visited += 1
        if nodes_visited > node_limit:
            limit_exceeded[0] = True
            return False
            
        if row_idx == n_rows:
            return True
        
        curr_len = len(current_row)
        if curr_len == k + 1:
            # Row complete
            diffs = {current_row[j] - current_row[i] for i in range(k+1) for j in range(i+1, k+1)}
            if diffs & used_diffs:
                return False
            
            # Pruning: Check if remaining rows can be filled
            rem_rows = n_rows - row_idx - 1
            needed = rem_rows * total_diffs_per_row
            available = max_possible_diffs - len(used_diffs) - len(diffs)
            if available < needed:
                return False
                
            rows.append(current_row)
            used_diffs.update(diffs)
            
            if solve_rows(row_idx + 1, [0], 0):
                return True
            
            used_diffs.difference_update(diffs)
            rows.pop()
            return False
        else:
            # Need next element
            rem_elements = k + 1 - curr_len
            max_val = M - (rem_elements - 1)
            
            # Pruning: Check if we can fit the remaining elements of this row
            # and all future rows
            # Current diffs in row: curr_len * (curr_len - 1) // 2
            # Diffs to be added in this row: total_diffs_per_row - curr_len * (curr_len - 1) // 2
            # Diffs in future rows: (n_rows - row_idx - 1) * total_diffs_per_row
            # Total needed: ...
            # Total available: max_possible_diffs - len(used_diffs)
            
            curr_diffs = curr_len * (curr_len - 1) // 2
            needed_curr = total_diffs_per_row - curr_diffs
            needed_future = (n_rows - row_idx - 1) * total_diffs_per_row
            total_needed = needed_curr + needed_future
            available = max_possible_diffs - len(used_diffs)
            
            if available < total_needed:
                return False
            
            for x in range(last_val + 1, max_val + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                # Additional pruning: 
                # The new diffs must not conflict with each other (they don't, as row is increasing)
                # And we check if the remaining space is sufficient after adding these diffs
                # This is implicit in the next call's check, but we can do a quick check here
                # If we pick x, the max element of this row will be at least x.
                
                if solve_rows(row_idx, current_row + [x], x):
                    return True
            return False
            
    if solve_rows(0, [0], 0):
        return rows
    return None
