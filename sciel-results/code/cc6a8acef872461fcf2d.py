import time

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 110.0 # Safety margin for 120s limit
    
    best_scope = float('inf')
    best_result = None
    
    # 1. Get a quick initial solution using a greedy approach to set best_scope
    # This provides a tight upper bound for the backtracking search.
    def greedy_solve():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last_val = 0
            while len(row) < k + 1:
                last_val += 1
                new_diffs = {last_val - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(last_val)
                    used_diffs.update(new_diffs)
            rows.append(row)
        return max(r[-1] for r in rows), rows

    initial_scope, initial_rows = greedy_solve()
    best_scope = initial_scope
    best_result = [r[:] for r in initial_rows]
    
    # 2. Systematic Backtracking with Branch and Bound
    # We try to find a solution with scope < best_scope.
    
    used_diffs = set()
    current_rows = []
    
    # Helper to check if a value x can be added to curr_row
    # Returns the set of new diffs if valid, else None
    def get_new_diffs(x, curr_row):
        diffs = set()
        for a in curr_row:
            d = x - a
            if d in used_diffs:
                return None
            diffs.add(d)
        return diffs
    
    def backtrack(row_idx, curr_row, curr_max):
        nonlocal best_scope, best_result, used_diffs, current_rows
        
        if time.time() - start_time > time_limit:
            return False
            
        if row_idx == n_rows:
            # Found a complete solution
            if curr_max < best_scope:
                best_scope = curr_max
                best_result = [r[:] for r in current_rows]
            return True
            
        if len(curr_row) == k + 1:
            # Row is complete, move to next row
            current_rows.append(curr_row)
            success = backtrack(row_idx + 1, [0], curr_max)
            current_rows.pop()
            return success
            
        last_val = curr_row[-1]
        # Pruning: The minimum possible max value for this row if we continue
        # is last_val + (k + 1 - len(curr_row))
        # If this exceeds best_scope, we can't improve the global best with this row.
        # Note: We are looking for scope < best_scope.
        # So if min_possible_max >= best_scope, prune.
        remaining_elements = k + 1 - len(curr_row)
        min_possible_max = last_val + remaining_elements
        
        if min_possible_max >= best_scope:
            return False
            
        # Iterate through possible next values
        # Start from last_val + 1
        # Upper bound for x in this row:
        # 1. Must be < best_scope (strictly, to improve)
        # 2. Must allow the remaining elements to fit < best_scope.
        #    If we pick x, the next is at least x+1, ..., last is at least x + (remaining_elements - 1).
        #    So x + (remaining_elements - 1) < best_scope => x < best_scope - (remaining_elements - 1)
        
        max_x = best_scope - remaining_elements
        
        # Also, x must be > last_val
        for x in range(last_val + 1, max_x + 1):
            new_diffs = get_new_diffs(x, curr_row)
            if new_diffs is None:
                continue
                
            # Apply choice
            curr_row.append(x)
            used_diffs.update(new_diffs)
            
            # Recurse
            # The new current max is x (since row is increasing)
            if backtrack(row_idx, curr_row, x):
                return True
                
            # Undo
            curr_row.pop()
            used_diffs.difference_update(new_diffs)
            
        return False
        
    # Start the backtracking search
    # We start with an empty set of rows and an empty used_diffs
    # We try to build the first row
    backtrack(0, [0], 0)
    
    # If backtracking didn't find anything better, we return the initial greedy result
    # which is stored in best_result.
    
    return best_result
