import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Generate all 5-subsets (Columns)
    points = range(12)
    all_5_subsets = list(combinations(points, 5))
    col_index = {s: i for i, s in enumerate(all_5_subsets)}
    num_cols = len(all_5_subsets) # 792
    
    # 2. Generate all 6-subsets (Rows)
    all_6_subsets = list(combinations(points, 6))
    rows = []
    for b in all_6_subsets:
        # Get indices of 5-subsets contained in b
        cols = []
        for s in combinations(b, 5):
            cols.append(col_index[s])
        rows.append(cols)
        
    # 3. Pre-compute: for each column, which rows cover it?
    # row_indices_for_col[col_idx] = list of row indices
    row_indices_for_col = [[] for _ in range(num_cols)]
    for r_idx, cols in enumerate(rows):
        for c in cols:
            row_indices_for_col[c].append(r_idx)
            
    # 4. Algorithm X Solver
    # State: available_cols (set of int), chosen_rows (list of int)
    
    def solve(available_cols, chosen_rows):
        if time.time() > deadline:
            return None # Timeout
            
        if not available_cols:
            return chosen_rows
            
        # Choose column with minimum active rows
        min_count = float('inf')
        target_col = -1
        active_rows_for_target = []
        
        for c in available_cols:
            count = 0
            # Check how many rows covering c are still fully available
            # A row is available if all its cols are in available_cols
            for r_idx in row_indices_for_col[c]:
                # Check if row r_idx is compatible
                # Optimized: just check if any col in rows[r_idx] is NOT in available_cols
                # Since rows are small (6 cols), this is fast
                if all(col in available_cols for col in rows[r_idx]):
                    count += 1
                    if count == min_count:
                        # Keep searching for smaller, or break if 1?
                        # If 1, it's a forced choice, break early?
                        pass
                        
            if count < min_count:
                min_count = count
                target_col = c
                # We need the list of active rows for this col to iterate
                # We will re-calculate or store it
                active_rows_for_target = [r_idx for r_idx in row_indices_for_col[c] if all(col in available_cols for col in rows[r_idx])]
                
            if min_count == 0:
                return None # Dead end
                
        # Iterate over active rows for target_col
        # Heuristic: Order rows? Maybe by how many cols they remove?
        # All remove 6.
        for r_idx in active_rows_for_target:
            # Remove cols covered by r_idx
            cols_to_remove = rows[r_idx]
            new_available = available_cols - set(cols_to_remove)
            
            result = solve(new_available, chosen_rows + [r_idx])
            if result is not None:
                return result
                
        return None

    initial_available = set(range(num_cols))
    solution_indices = solve(initial_available, [])
    
    if solution_indices is None:
        # Fallback to greedy if solver fails/times out
        # (Unlikely for this specific problem)
        return []
        
    blocks = [list(all_6_subsets[i]) for i in solution_indices]
    return blocks
