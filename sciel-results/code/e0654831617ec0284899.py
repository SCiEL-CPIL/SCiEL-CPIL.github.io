   def dfs(row_idx, pos, current_row, used_diffs, M):
       if row_idx == n_rows:
           return True
       if pos == k + 1:
           # Row complete, move to next
           return dfs(row_idx + 1, 0, [0], used_diffs, M)
       
       # Determine candidate range
       # If pos == 0, x is fixed to 0? No, row starts at 0.
       # So for pos=0, x=0.
       # For pos > 0, x must be > current_row[-1] and <= M.
       
       if pos == 0:
           # x = 0
           # No new diffs
           return dfs(row_idx, pos + 1, [0], used_diffs, M)
       
       last = current_row[-1]
       # Candidates: last + 1 to M
       # Heuristic: Try candidates that create "common" differences? No, we want distinct.
       # Try candidates that are "sparse"?
       
       # Optimization: Check if there are enough unused differences available in the range [1, M]
       # to fill the remaining slots for this row and subsequent rows.
       
       # Simple loop
       for x in range(last + 1, M + 1):
           new_diffs = {x - a for a in current_row}
           if new_diffs & used_diffs:
               continue
           
           # Recurse
           if dfs(row_idx, pos + 1, current_row + [x], used_diffs | new_diffs, M):
               return True
           
           # Backtrack: remove new_diffs from used? 
           # We passed a new set, so no need to modify in place if we use immutable sets.
           # But creating sets is slow.
           # Let's use a mutable set and add/remove.
           
       return False
   