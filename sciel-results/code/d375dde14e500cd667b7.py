import time
import sys

def construct(n_rows, k):
    # Start time
    start_time = time.time()
    deadline = start_time + 100.0  # 100 seconds limit
    
    # Lower bound on L
    # Total differences needed: n_rows * k * (k+1) / 2
    # Max difference is L
    # So L must be at least count
    min_L = n_rows * k * (k + 1) // 2
    
    # Get a greedy solution to have a fallback and an upper bound
    greedy_rows = greedy_construct(n_rows, k)
    current_best_L = max(max(row) for row in greedy_rows)
    best_rows = greedy_rows
    
    # Try to improve by searching for smaller L
    # We search L from min_L up to current_best_L
    # If we find a valid set for L, we update best_rows and current_best_L
    # We can stop if we reach min_L
    
    # To make it faster, we might want to search downwards if we have a good heuristic, 
    # but searching upwards from the lower bound ensures we find the optimal if we get there.
    # However, the space is huge. 
    # A better strategy: Binary search on L? No, validity is not monotonic in a simple way for *existence*? 
    # Actually, if a set exists for L, it exists for L+1 (just don't use the new marks). 
    # So validity is monotonic. We can binary search for the minimum L.
    
    low = min_L
    high = current_best_L
    
    # If the gap is small, linear search might be faster.
    # Let's try linear search from low upwards, but with a strict time limit.
    # If we don't find a solution for L, we try L+1.
    
    # Optimization: Use a better search order for the first row.
    # The first row defines the "structure".
    
    for L in range(low, high + 1):
        if time.time() > deadline:
            break
            
        # Check if a solution exists for this L
        result = backtracking_search(n_rows, k, L, start_time, deadline)
        
        if result is not None:
            best_rows = result
            current_best_L = L
            # Since we are searching upwards from the lower bound, 
            # the first valid L we find is the minimum L.
            # So we can return immediately?
            # Yes, because we want the *smallest* L.
            return best_rows
            
    return best_rows

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

def backtracking_search(n_rows, k, L, start_time, deadline):
    # used_diffs is a set of integers
    # We construct rows one by one.
    # To reduce symmetry, we can assume the first row is "minimal" in some sense?
    # For now, just DFS.
    
    # State: 
    # - current row index
    # - current row marks
    # - used differences
    
    # We need to be careful with recursion depth and time.
    
    def dfs(row_idx, current_row, used_diffs):
        if time.time() > deadline:
            return None
            
        if row_idx == n_rows:
            return [current_row] # Actually we need to return all rows
            
        # If current_row is full, move to next row
        if len(current_row) == k + 1:
            # Start next row
            # Optimization: The next row must also fit in [0, L]
            # We can try to construct the next row
            # Note: We need to keep track of all rows to return them
            # Let's pass a list of completed rows
            
            # This structure is getting complicated. Let's restructure.
            
            pass

    # Let's use a different structure.
    # rows = []
    # used = set()
    # def solve_row(idx):
    #   if idx == n_rows: return True
    #   row = [0]
    #   def solve_mark(pos):
    #     if pos == k+1:
    #       rows.append(row)
    #       if solve_row(idx+1): return True
    #       rows.pop()
    #       return False
    #     # try marks from row[-1]+1 to L
    #     ...
    #   return solve_mark(1)
    
    # This is cleaner.
    
    rows = []
    used_diffs = set()
    
    def solve_row(idx):
        if idx == n_rows:
            return True
        if time.time() > deadline:
            return False
            
        row = [0]
        
        def solve_mark(pos):
            if pos == k + 1:
                rows.append(row)
                if solve_row(idx + 1):
                    return True
                rows.pop()
                return False
                
            last = row[-1]
            # Try next marks
            # Optimization: We can limit the range.
            # The maximum possible mark is L.
            for x in range(last + 1, L + 1):
                # Check time
                if time.time() > deadline:
                    return False
                    
                # Check differences
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                    
                # Place mark
                row.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                
                if solve_mark(pos + 1):
                    return True
                    
                # Backtrack
                for d in new_diffs:
                    used_diffs.remove(d)
                row.pop()
                
            return False
            
        return solve_mark(1)
        
    if solve_row(0):
        return rows
    return None
