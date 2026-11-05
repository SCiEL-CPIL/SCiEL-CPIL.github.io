import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115  # Leave some buffer
    
    # Lower bound: counting argument
    # Number of differences per row: k*(k+1)/2
    # Total differences: n_rows * k*(k+1)/2
    # These differences must be distinct and <= M
    # So M >= n_rows * k*(k+1)/2
    # For n_rows=4, k=7: 4 * 28 = 112.
    lower_bound = n_rows * k * (k + 1) // 2
    
    # Upper bound: Greedy construction
    def greedy_init():
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
                    x += 1
                else:
                    x += 1
            rows.append(row)
        return rows

    init_rows = greedy_init()
    upper_bound = max(max(r) for r in init_rows)
    
    # If greedy is already good, maybe just return it if we run out of time?
    # But we want to improve.
    
    # Binary search for the smallest M
    # We will try to find a solution with max <= M
    
    # To speed up DFS, we can use a heuristic for row ordering?
    # Or just standard DFS.
    
    # We need to store the best solution found so far (from greedy)
    best_solution = init_rows
    
    def can_construct(M):
        """
        DFS to check if a valid set exists with max value <= M.
        Returns the set of rows if found, else None.
        """
        used_diffs = set()
        rows = []
        
        # Helper to check if adding 'val' to 'current_row' is valid
        def is_valid(val, current_row, used_diffs):
            # Check internal conflicts
            # Differences with previous elements in current_row
            for prev in current_row:
                d = val - prev
                if d in used_diffs:
                    return False
            # We don't need to check internal conflicts of current_row against itself 
            # if we build it sequentially and check against used_diffs which 
            # includes diffs from previous rows. 
            # Wait, used_diffs should contain ALL diffs from ALL previous rows 
            # AND the diffs from the current row so far?
            # If we update used_diffs as we add elements to current_row, 
            # then 'used_diffs' contains diffs from previous rows + current row so far.
            # So checking 'd in used_diffs' covers both cross-row and internal (with prev elements).
            
            # Also need to ensure strictly increasing
            if current_row and val <= current_row[-1]:
                return False
                
            return True

        def dfs(row_idx, col_idx, current_row):
            # Check time
            if time.time() - start_time > time_limit:
                return None
            
            # Base case: All rows constructed
            if row_idx == n_rows:
                return [list(r) for r in rows]
            
            # If we are starting a new row
            if col_idx == 0:
                # Initialize current row with 0
                current_row = [0]
                # 0 creates no positive differences, so no update to used_diffs needed for the element itself
                # But we need to proceed to col_idx 1
                return dfs(row_idx, 1, current_row)
            
            # If we have finished the current row (length k+1)
            if col_idx == k + 1:
                # Add this row to the list of completed rows
                rows.append(current_row)
                # Recurse to next row
                result = dfs(row_idx + 1, 0, [])
                if result is not None:
                    return result
                # Backtrack
                rows.pop()
                return None
            
            # Determine range for current_row[col_idx]
            # Lower bound: previous element + 1
            lower = current_row[-1] + 1
            # Upper bound: M
            # Also, we need to leave room for the remaining (k - col_idx) elements
            # Max possible value for current element if we want to fit strictly increasing to M
            # val + 1 + ... + (k - col_idx) <= M
            # val <= M - (k - col_idx) * (k - col_idx + 1) / 2 ? No, just M - (k - col_idx) is a loose bound.
            # Actually, if we are at index i (0-indexed, so i=col_idx-1 in 0-based list), 
            # we need k+1 - (col_idx+1) more elements.
            # Let remaining = k - col_idx
            # val + 1 + ... + remaining <= M => val + remaining*(remaining+1)/2 <= M
            remaining = k - col_idx
            upper = M - (remaining * (remaining + 1)) // 2
            if upper < lower:
                return None
            
            # Iterate through possible values
            # Heuristic: Try smaller values first to keep max low? 
            # Or try values that are "sparse"?
            # Standard is increasing order.
            for val in range(lower, upper + 1):
                # Check validity
                if is_valid(val, current_row, used_diffs):
                    # Calculate new differences
                    new_diffs = set()
                    for prev in current_row:
                        new_diffs.add(val - prev)
                    
                    # Check if any new_diff is in used_diffs
                    # is_valid already checked this, but let's be sure
                    # Actually is_valid checked 'd in used_diffs' for each prev.
                    # So we are safe.
                    
                    # Apply move
                    current_row.append(val)
                    used_diffs.update(new_diffs)
                    
                    # Recurse
                    result = dfs(row_idx, col_idx + 1, current_row)
                    if result is not None:
                        return result
                    
                    # Backtrack
                    used_diffs.difference_update(new_diffs)
                    current_row.pop()
            
            return None
        
        return dfs(0, 0, [])

    # Binary Search for the minimum M
    low = lower_bound
    high = upper_bound
    
    # To save time, we can first try to improve the greedy solution slightly
    # or just binary search.
    
    # If the range is small, linear search might be faster, but binary is safe.
    # We want the *smallest* M that works.
    
    # Optimization: If we find a solution, we can stop if M == lower_bound.
    
    # Let's iterate M from lower_bound upwards? 
    # No, binary search is better if the space is large.
    # But for small n, the gap between lower and upper is small (112 vs ~134).
    # So linear search from lower_bound to upper_bound is very fast and guarantees finding the min.
    
    # Let's do linear search from lower_bound to upper_bound.
    # This ensures we find the absolute minimum M in this range.
    
    found_solution = None
    for M in range(lower_bound, upper_bound + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try to construct with max <= M
        sol = can_construct(M)
        if sol is not None:
            # Check if this is better than our current best
            current_max = max(max(r) for r in best_solution)
            if M < current_max:
                best_solution = sol
                # We found a smaller M. Since we are iterating upwards, 
                # this is the smallest M possible >= lower_bound.
                # We can break here because we are looking for the minimum M.
                break
        else:
            # No solution for this M, try next
            continue
            
    # If no solution found in the range (should not happen if upper_bound is valid),
    # return the greedy solution.
    
    return best_solution
