import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    timeout = 145.0  # Safety margin against 150s kill
    k_plus_1 = k + 1
    
    # Helper to check validity of a single row against used mask
    def is_row_valid(row, used_mask):
        new_diffs = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if (used_mask >> d) & 1:
                    return False
                if (new_diffs >> d) & 1:
                    return False
                new_diffs |= (1 << d)
        return True

    # Helper to get the diff mask of a row
    def get_row_mask(row):
        mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    # Greedy to get an initial upper bound
    def greedy_construct():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k_plus_1:
                x = row[-1] + 1
                while True:
                    new_diffs = 0
                    valid = True
                    for a in row:
                        d = x - a
                        if (used >> d) & 1:
                            valid = False
                            break
                        if (new_diffs >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    if valid:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_max = max(max(r) for r in best_rows)
    
    # Binary search on the maximum value M
    # Lower bound: counting bound
    total_diffs_needed = n_rows * (k_plus_1 * (k_plus_1 - 1) // 2)
    low = total_diffs_needed
    high = best_max
    
    def solve_with_max(M):
        """
        Returns a list of rows if a valid DTS exists with all elements <= M, else None.
        Uses DFS with symmetry breaking (rows sorted by max).
        """
        used_mask = 0
        rows = []
        
        # Symmetry breaking: enforce max(row_i) <= max(row_{i+1})
        # We track the max of the previous row
        prev_max = 0
        
        def dfs(row_idx, current_row, used_mask, prev_max):
            nonlocal best_rows, best_max
            
            if time.time() - start_time > timeout:
                return False
            
            if row_idx == n_rows:
                # Found a solution
                return True
            
            # If current_row is complete, we have a new row
            # We need to decide if we are building a row or starting a new one?
            # The structure is: build row 0, then row 1, etc.
            # So `current_row` is the row currently being built.
            
            if len(current_row) == k_plus_1:
                # Row is complete. Check if it satisfies symmetry constraint
                row_max = current_row[-1]
                if row_max < prev_max:
                    return False # Violates symmetry
                
                new_used_mask = used_mask | get_row_mask(current_row)
                # Recurse to next row
                if dfs(row_idx + 1, [0], new_used_mask, row_max):
                    return True
                return False
            
            # Otherwise, add next element to current_row
            # Candidates: from current_row[-1] + 1 to M
            # Also, we must ensure we can complete the row within M
            slots_left = k_plus_1 - len(current_row)
            # Minimum possible next values: current_row[-1] + 1, ..., current_row[-1] + slots_left
            # If current_row[-1] + slots_left > M, we can't complete this row.
            if current_row[-1] + slots_left > M:
                return False
            
            start_val = current_row[-1] + 1
            # We iterate in increasing order to find the "tightest" fit first
            for x in range(start_val, M + 1):
                # Check if x creates valid differences
                new_diffs = 0
                valid = True
                for a in current_row:
                    d = x - a
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    if (new_diffs >> d) & 1:
                        valid = False
                        break
                    new_diffs |= (1 << d)
                
                if not valid:
                    continue
                
                # Pruning: If we add x, the max of this row will be at least x.
                # If this row is not the last one, its max must be <= next rows' max?
                # We enforce max(row_i) <= max(row_{i+1}).
                # So if we are building row i, and we pick x, the final max will be >= x.
                # This doesn't help prune much unless we know the max of subsequent rows.
                
                if dfs(row_idx, current_row + [x], used_mask | new_diffs, prev_max):
                    # Update best if this is a better solution?
                    # Actually, we are just checking if a solution exists for M.
                    # If it exists, we return it.
                    # We need to capture the rows.
                    # The `dfs` function modifies `rows`? No, it's pure recursion.
                    # We need to reconstruct the solution.
                    # This is inefficient.
                    
                    # Better approach:
                    # Instead of returning True/False, we can store the path.
                    # Or, since we are binary searching M, we just need ONE solution.
                    # We can modify `dfs` to append to a global `found_rows` list.
                    
                    pass # Placeholder for logic below
            
            return False
        
        # Re-implement DFS to store the solution
        found_rows = []
        
        def dfs_store(row_idx, current_row, used_mask, prev_max):
            nonlocal found_rows
            
            if time.time() - start_time > timeout:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k_plus_1:
                row_max = current_row[-1]
                if row_max < prev_max:
                    return False
                
                new_used_mask = used_mask | get_row_mask(current_row)
                found_rows.append(current_row)
                
                if dfs_store(row_idx + 1, [0], new_used_mask, row_max):
                    return True
                
                found_rows.pop()
                return False
            
            slots_left = k_plus_1 - len(current_row)
            if current_row[-1] + slots_left > M:
                return False
            
            start_val = current_row[-1] + 1
            for x in range(start_val, M + 1):
                new_diffs = 0
                valid = True
                for a in current_row:
                    d = x - a
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    if (new_diffs >> d) & 1:
                        valid = False
                        break
                    new_diffs |= (1 << d)
                
                if not valid:
                    continue
                
                if dfs_store(row_idx, current_row + [x], used_mask | new_diffs, prev_max):
                    return True
            
            return False
        
        if dfs_store(0, [0], 0, 0):
            return found_rows
        return None

    # Binary Search Loop
    # We want the smallest M such that solve_with_max(M) is not None.
    # Start with high = best_max (known valid).
    # We can try to lower M.
    
    # Optimization: If the greedy solution is already close to the bound, 
    # the binary search range is small.
    
    # We will iterate M downwards from best_max? 
    # Binary search is better if the range is large.
    # But solve_with_max is expensive.
    # Let's try a few values near the bound.
    
    # The counting bound is `low`.
    # We know `high` is valid.
    
    # Strategy: Try M = low, low+1, ... up to high?
    # Or Binary Search?
    # If we use Binary Search, we might miss the exact minimum if the function is not monotonic?
    # It IS monotonic: If a solution exists for M, it exists for M+1 (just don't use the extra space).
    # So Binary Search is valid.
    
    left = low
    right = best_max
    best_found = best_rows
    best_m = best_max
    
    while left <= right and time.time() - start_time < timeout:
        mid = (left + right) // 2
        res = solve_with_max(mid)
        if res is not None:
            best_found = res
            best_m = mid
            right = mid - 1
        else:
            left = mid + 1
            
    return best_found
# EVOLVE-BLOCK-END
