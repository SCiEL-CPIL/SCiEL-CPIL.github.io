import time
import random

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set with n_rows rows and k+1 columns.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # 1. Get an initial solution using a greedy approach to set an upper bound
    def greedy_construct():
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

    initial_rows = greedy_construct()
    best_scope = max(max(row) for row in initial_rows)
    best_rows = initial_rows
    
    # 2. Backtracking search to improve the scope
    # State:
    # rows: list of completed rows
    # used_diffs: set of all differences
    # current_row: list of ints for the row being built
    # row_idx: index of current row
    # col_idx: index of current column in row
    
    # We will try to find a solution with max value < best_scope
    # We use a global variable to track the best found during search
    search_best_scope = best_scope
    search_best_rows = best_rows
    nodes_visited = 0
    time_exceeded = False
    
    def backtrack(rows, used_diffs, current_row, row_idx, col_idx):
        nonlocal search_best_scope, search_best_rows, nodes_visited, time_exceeded
        
        if time_exceeded:
            return
        
        if time.time() - start_time > time_limit:
            time_exceeded = True
            return
            
        nodes_visited += 1
        
        # If we have completed all rows
        if row_idx == n_rows:
            if col_idx == 0:
                # Check if this is better
                current_scope = max(max(r) for r in rows)
                if current_scope < search_best_scope:
                    search_best_scope = current_scope
                    search_best_rows = [r[:] for r in rows]
            return
        
        # If we have completed the current row, move to next
        if col_idx == k + 1:
            backtrack(rows + [current_row], used_diffs, [], row_idx + 1, 0)
            return
        
        # Determine the minimum value for the next element
        min_val = current_row[-1] + 1 if current_row else 1
        
        # Pruning: Check if we can even fit the remaining columns within search_best_scope
        # We need (k + 1 - col_idx) more elements.
        # The smallest possible sequence is min_val, min_val+1, ...
        # So the largest will be min_val + (k - col_idx)
        # If this exceeds search_best_scope, prune.
        if min_val + (k - col_idx) > search_best_scope:
            return
            
        # Try values from min_val to search_best_scope
        # Optimization: We can break early if we find a good value, but we need to explore all to find the global min?
        # Actually, since we are minimizing the max, and we are iterating in increasing order,
        # the first complete solution found with a given scope is not necessarily the global min,
        # but we are updating search_best_scope dynamically.
        
        for v in range(min_val, search_best_scope + 1):
            if time.time() - start_time > time_limit:
                time_exceeded = True
                return
                
            # Check differences with current row elements
            new_diffs = set()
            conflict = False
            for a in current_row:
                d = v - a
                if d in used_diffs or d in new_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                continue
                
            # Recursive step
            backtrack(rows, used_diffs | new_diffs, current_row + [v], row_idx, col_idx + 1)
            
            # If we found a better solution, we might prune further?
            # If search_best_scope has decreased, the loop condition `v < search_best_scope` will naturally adjust.
            # But we are iterating `range(min_val, search_best_scope + 1)`.
            # If search_best_scope changes, the range object is already created.
            # We should check if v is still within the new bound?
            # Actually, if we found a solution with scope S, any branch with max > S is useless.
            # Since we are building increasing rows, the max of the final solution is at least v.
            # If v > search_best_scope, we can break.
            if v > search_best_scope:
                break

    # Start the search
    # Initial state: no rows, no diffs, empty current row, row 0, col 0
    backtrack([], set(), [], 0, 0)
    
    return search_best_rows

# EVOLVE-BLOCK-END
