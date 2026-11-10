import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115
    
    # Lower bound for scope
    # Total diffs = n_rows * k * (k+1) / 2
    total_diffs = n_rows * k * (k + 1) // 2
    min_scope = total_diffs
    
    # We know best known is 112 for (7,5).
    # Let's try to find a solution with scope close to min_scope.
    
    def solve_with_scope(scope):
        """
        Try to find a solution with max entry <= scope.
        Returns list of rows if found, else None.
        """
        used = set()
        rows = []
        
        def dfs(row_idx, current_row):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                return dfs(row_idx + 1, [0])
            
            # Pruning: Check if we have enough room for remaining diffs
            # Diffs used so far in this row: len(current_row)*(len(current_row)-1)//2
            # Diffs in full rows: row_idx * k * (k+1) // 2
            # Total diffs needed: total_diffs
            # If used.size() + (remaining full rows)*diffs_per_row + (current row partial diffs) > scope, prune?
            # Actually, just check if len(used) + (needed_diffs) > scope is a weak check.
            # Stronger: The set of available diffs in [1, scope] must be able to accommodate the rest.
            
            # Heuristic: Try next numbers
            # The next number must be > current_row[-1]
            start_val = current_row[-1] + 1
            
            # Optimization: We can limit the search space for the next element.
            # It must be <= scope.
            
            # To improve speed, we can randomize the order of candidates slightly or use a heuristic.
            # However, for small k, linear scan is fast.
            
            for x in range(start_val, scope + 1):
                # Check if x creates valid new diffs
                new_diffs = {x - a for a in current_row}
                if new_diffs & used:
                    continue
                
                # Additional Pruning: 
                # If we pick x, the new used set is used | new_diffs.
                # We need to ensure that the remaining rows can be filled.
                # A simple necessary condition:
                # Number of available diffs in [1, scope] not in used must be >= required diffs.
                # Required diffs for current row completion:
                # Current row has L elements. We need K+1-L more.
                # Each new element adds L, L+1, ... diffs.
                # This is complex to compute exactly on the fly.
                # Let's stick to the basic check and rely on backtracking.
                
                # Try x
                used.update(new_diffs)
                current_row.append(x)
                
                if dfs(row_idx, current_row):
                    return True
                
                # Backtrack
                current_row.pop()
                used.difference_update(new_diffs)
                
            return False
        
        # Start with first row [0]
        if dfs(0, [0]):
            return rows
        return None

    # Iterate over possible scopes
    # Start from min_scope, go up to some limit
    # If we find a solution, return it.
    # To handle time limits, we can interleave or just stop if time is up.
    
    # Since the problem is small (7 rows, 6 cols), a simple DFS might be slow if it explores too many branches.
    # Let's add a timeout check inside DFS.
    
    # We can also try random restarts with a fixed scope if DFS fails quickly.
    
    best_rows = None
    best_scope = float('inf')
    
    # Try increasing scopes
    # For (7,5), min_scope is 105. Best is 112.
    # Let's try 105, 106, ..., up to 120 or until time runs out.
    
    # To make it faster, we can use a heuristic in DFS:
    # Prefer smaller x (to keep scope low) but this is already done by iterating x from start_val.
    # However, iterating x from start_val to scope is O(scope).
    
    # Improvement: 
    # If DFS for a specific scope is too slow, we might need a better heuristic.
    # For now, let's try this.
    
    for scope in range(min_scope, min_scope + 50): # Search up to 50 above lower bound
        if time.time() > deadline - 5:
            break
            
        rows = solve_with_scope(scope)
        if rows:
            return rows
            
    # If no solution found within the loop (due to time or failure), 
    # fall back to the greedy approach which is guaranteed to finish fast.
    
    # Greedy fallback
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
