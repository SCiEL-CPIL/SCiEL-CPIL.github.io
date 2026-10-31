import sys

def construct(n_rows, k):
    # Calculate lower bound for M
    # Total differences = n_rows * k * (k+1) / 2
    total_diffs = n_rows * (k * (k + 1)) // 2
    M = total_diffs
    
    # If M is very small, we might need to increase it.
    # The lower bound is a lower bound on the max difference, which is <= M.
    # Actually, the max difference in a row is <= M.
    # The differences are in [1, M].
    # So M must be at least total_diffs.
    
    while True:
        used_diffs = set()
        
        def solve(row_idx, current_row, rows_so_far):
            if len(current_row) == k + 1:
                if row_idx == n_rows - 1:
                    return rows_so_far + [current_row]
                return solve(row_idx + 1, [0], rows_so_far + [current_row])
            
            last_val = current_row[-1]
            # Optimization: The next value x must allow for the remaining (k - len(current_row)) values
            # to be placed in (x, M].
            # The minimum number of steps is 1 each, so we need M - x >= (k - len(current_row))
            # Thus x <= M - (k - len(current_row))
            max_x = M - (k - len(current_row))
            
            for x in range(last_val + 1, max_x + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                used_diffs.update(new_diffs)
                
                result = solve(row_idx, current_row + [x], rows_so_far)
                
                if result is not None:
                    return result
                
                used_diffs.difference_update(new_diffs)
            
            return None
        
        solution = solve(0, [0], [])
        if solution is not None:
            return solution
        
        M += 1
