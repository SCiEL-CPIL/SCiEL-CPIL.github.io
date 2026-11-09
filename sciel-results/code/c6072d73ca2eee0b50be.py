import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin for 120s limit
    
    n_points = 12
    k_block = 6
    k_sub = 5
    
    # 1. Generate all 5-subsets and assign indices
    all_5_subsets = list(itertools.combinations(range(n_points), k_sub))
    num_5_subsets = len(all_5_subsets)
    # Map 5-subset to index
    # Using a dictionary for O(1) lookup
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # 2. Generate all 6-subsets and their corresponding 5-subset masks
    all_6_subsets = list(itertools.combinations(range(n_points), k_block))
    num_6_subsets = len(all_6_subsets)
    
    # row_masks[i] is a bitset of 5-subsets covered by 6-subset i
    row_masks = [0] * num_6_subsets
    # col_to_rows[j] is a list of row indices that cover 5-subset j
    col_to_rows = [[] for _ in range(num_5_subsets)]
    
    for i, block in enumerate(all_6_subsets):
        mask = 0
        # A 6-subset contains 6 5-subsets
        for j in range(k_block):
            # Remove element at index j to form a 5-subset
            # To do this efficiently, we can just iterate combinations of 5 from 6
            pass
        
        # Better: iterate combinations of 5 from the 6 elements
        for sub5 in itertools.combinations(block, k_sub):
            idx = subset_to_idx[sub5]
            mask |= (1 << idx)
            col_to_rows[idx].append(i)
            
        row_masks[i] = mask

    # 3. Solve Exact Cover using Algorithm X (Backtracking with Minimum Remaining Values)
    full_mask = (1 << num_5_subsets) - 1
    
    # Precompute candidate counts for initial heuristic
    # We will pick the column with the fewest available rows
    
    solution = []
    
    def solve(covered_mask, depth):
        if time.time() > deadline:
            return False
            
        if covered_mask == full_mask:
            return True
            
        # Find the uncovered column with the minimum number of candidates
        # We can iterate through all columns, but that's 792 checks.
        # Optimization: Keep track of counts? 
        # For simplicity and speed in Python, iterating 792 columns is fast enough (~microseconds).
        
        min_count = num_6_subsets + 1
        best_col = -1
        best_candidates = []
        
        # Iterate through all columns
        for col in range(num_5_subsets):
            if covered_mask & (1 << col):
                continue
                
            candidates = col_to_rows[col]
            # Filter candidates that don't conflict with covered_mask
            valid_candidates = []
            for row in candidates:
                if row_masks[row] & covered_mask:
                    continue
                valid_candidates.append(row)
                
            count = len(valid_candidates)
            if count < min_count:
                min_count = count
                best_col = col
                best_candidates = valid_candidates
                
            if min_count == 0:
                return False # Dead end
                
        if min_count == 0:
            return False

        # Try each candidate in best_candidates
        for row in best_candidates:
            # Select row
            new_covered = covered_mask | row_masks[row]
            solution.append(all_6_subsets[row])
            
            if solve(new_covered, depth + 1):
                return True
                
            solution.pop()
            
        return False

    if solve(0, 0):
        return solution
    else:
        # If no solution found (should not happen), return empty or best effort
        return []
