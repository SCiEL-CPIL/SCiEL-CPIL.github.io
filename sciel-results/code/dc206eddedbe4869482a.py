import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 integers starting at 0, strictly increasing,
    such that all differences are distinct globally.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    # 1. Greedy Initialization
    def greedy_init():
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

    current_rows = greedy_init()
    best_rows = [r[:] for r in current_rows]
    
    def get_global_max(rows):
        return max(max(r) for r in rows)
    
    current_global_max = get_global_max(current_rows)
    best_global_max = current_global_max
    
    # Helper to compute diffs for a row
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    # Precompute diffs for other rows
    def get_other_diffs(rows, r):
        diffs = set()
        for i in range(n_rows):
            if i != r:
                diffs |= get_diffs(rows[i])
        return diffs

    # DFS to reconstruct a row with max <= limit
    # Returns the row or None
    def reconstruct_row(other_diffs, limit, row_len):
        # Use recursion with backtracking
        # To avoid recursion limit issues, we can use a stack or increase limit, but row_len is small (9)
        # We'll use a helper that tries to fill the row
        row = [0]
        row_diffs = set()
        
        def dfs(current_row, current_diffs, next_x):
            if len(current_row) == row_len:
                return current_row[:]
            
            # Try values from next_x to limit
            # Optimization: We can skip values that are too small? No, we want smallest.
            # We can also prune if next_x - current_row[0] > limit? No, next_x is the value.
            
            # Heuristic: Try smaller x first to keep max low.
            for x in range(next_x, limit + 1):
                new_diffs = {x - a for a in current_row}
                # Check conflict with other rows
                if new_diffs & other_diffs:
                    continue
                # Check conflict within this row (already in current_diffs)
                if new_diffs & current_diffs:
                    continue
                
                # Recurse
                current_row.append(x)
                current_diffs |= new_diffs
                
                result = dfs(current_row, current_diffs, x + 1)
                if result is not None:
                    return result
                
                # Backtrack
                current_row.pop()
                current_diffs -= new_diffs
                
            return None
        
        return dfs(row, row_diffs, 1)

    # Main Loop
    while time.time() < deadline:
        r = random.randint(0, n_rows - 1)
        other_diffs = get_other_diffs(current_rows, r)
        
        # We want to find a row with max <= current_global_max
        # Ideally, we want to reduce the global max.
        # If row r is the one with the max, we try to reduce it.
        # If not, we try to keep it <= current_global_max.
        
        target_limit = current_global_max
        
        # Try to reconstruct
        new_row = reconstruct_row(other_diffs, target_limit, k + 1)
        
        if new_row is not None:
            new_max = max(new_row)
            # Calculate new global max
            other_max = max(max(current_rows[i]) for i in range(n_rows) if i != r)
            new_global_max = max(new_max, other_max)
            
            if new_global_max < best_global_max:
                best_global_max = new_global_max
                best_rows = [row[:] for row in current_rows]
                best_rows[r] = new_row
                current_rows[r] = new_row
                current_global_max = new_global_max
            else:
                # Accept with some probability to allow exploration
                # If new_max < current_rows[r][-1], it's a strict improvement for that row
                if random.random() < 0.3:
                    current_rows[r] = new_row
                    current_global_max = new_global_max
        
        # Optional: If we found a better best, we can try to lower the limit for future searches
        # But current_global_max is already updated if we accepted a better one.
        
    return best_rows

# EVOLVE-BLOCK-END
