import sys

def construct(n_rows, k):
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound for max value
    # The differences must be distinct and positive.
    # The largest difference is at least total_diffs.
    # Also, the largest entry M must allow for these differences.
    # A simple lower bound is total_diffs.
    # A better lower bound: The sum of differences in a row is not directly constrained, 
    # but the max difference in a row is <= M.
    # Let's just start from total_diffs.
    
    lower_bound = total_diffs
    # Upper bound: Let's estimate from a greedy run or just set a reasonable limit.
    # The optimal for (7,5) is 112. Let's search from 105 to 120.
    
    for M in range(lower_bound, lower_bound + 20):
        if can_construct(n_rows, k, M):
            # Re-run to get the actual rows, or store them during the check.
            # To be safe and simple, we re-run with a solver that returns the rows.
            return solve(n_rows, k, M)
            
    # Fallback to greedy if backtracking fails or times out (should not happen for small N)
    return greedy_fallback(n_rows, k)

def can_construct(n_rows, k, M):
    used_diffs = 0
    rows = []
    return backtrack_rows(0, rows, used_diffs, n_rows, k, M)

def backtrack_rows(row_idx, rows, used_diffs, n_rows, k, M):
    if row_idx == n_rows:
        return True
    
    # Try to construct the next row
    row = []
    if backtrack_row(0, row, used_diffs, k, M):
        rows.append(row)
        # Update used_diffs
        new_used = used_diffs
        for i in range(len(row)):
            for j in range(i):
                diff = row[i] - row[j]
                new_used |= (1 << diff)
        
        if backtrack_rows(row_idx + 1, rows, new_used, n_rows, k, M):
            return True
        
        # Backtrack: remove row
        rows.pop()
        # Note: used_diffs is passed by value (int), so no need to un-set bits here 
        # if we passed it correctly. But we modified new_used locally.
        # The recursive call used new_used. If it failed, we just return.
    return False

def backtrack_row(index, row, used_diffs, k, M):
    # row currently has 'index' elements (0 to index-1)
    # We need to add k+1 elements total. So we need to add up to index == k+1.
    if index == k + 1:
        return True
    
    last_val = row[-1] if row else 0
    # Try next values
    # Optimization: We can limit the range of x.
    # x must be > last_val and <= M.
    # Also, the differences x - a_i must be available.
    
    # Heuristic: Try smaller x first? Or larger?
    # Smaller x tends to keep the ruler compact.
    
    for x in range(last_val + 1, M + 1):
        # Check if differences x - a_i are available
        conflict = False
        for a in row:
            diff = x - a
            if diff > M: # Should not happen if x <= M and a >= 0, but diff can be large? No, max diff is M.
                break
            if (used_diffs & (1 << diff)):
                conflict = True
                break
        
        if not conflict:
            row.append(x)
            # Update used_diffs for this step? 
            # It's better to pass the updated used_diffs to the next recursion level
            # to avoid modifying and reverting.
            new_used = used_diffs
            for a in row[:-1]: # Exclude x itself from diff calculation? 
                # Wait, the differences are between x and all previous elements in row.
                # The loop above checked availability. Now we mark them as used.
                pass
            
            # Let's just calculate the new used bits
            new_used = used_diffs
            for a in row[:-1]:
                diff = x - a
                new_used |= (1 << diff)
            
            if backtrack_row(index + 1, row, new_used, k, M):
                return True
            
            row.pop()
            
    return False

def greedy_fallback(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def solve(n_rows, k, M):
    used_diffs = 0
    rows = []
    # We know can_construct returned True, so this should succeed quickly.
    # We can use the same backtracking but it needs to return the rows.
    # Since can_construct already did the work, we can just store the rows in the global or pass a list.
    
    # Let's modify can_construct to return the rows if found.
    # For simplicity, I'll just re-run the search with a flag to collect rows.
    
    return search_and_collect(n_rows, k, M)

def search_and_collect(n_rows, k, M):
    used_diffs = 0
    rows = []
    # Re-implement backtrack to collect
    if collect_backtrack_rows(0, rows, used_diffs, n_rows, k, M):
        return rows
    return []

def collect_backtrack_rows(row_idx, rows, used_diffs, n_rows, k, M):
    if row_idx == n_rows:
        return True
    
    row = []
    if collect_backtrack_row(0, row, used_diffs, k, M):
        rows.append(row)
        new_used = used_diffs
        for i in range(len(row)):
            for j in range(i):
                diff = row[i] - row[j]
                new_used |= (1 << diff)
        
        if collect_backtrack_rows(row_idx + 1, rows, new_used, n_rows, k, M):
            return True
        
        rows.pop()
    return False

def collect_backtrack_row(index, row, used_diffs, k, M):
    if index == k + 1:
        return True
    
    last_val = row[-1] if row else 0
    for x in range(last_val + 1, M + 1):
        conflict = False
        for a in row:
            diff = x - a
            if (used_diffs & (1 << diff)):
                conflict = True
                break
        
        if not conflict:
            row.append(x)
            new_used = used_diffs
            for a in row[:-1]:
                diff = x - a
                new_used |= (1 << diff)
            
            if collect_backtrack_row(index + 1, row, new_used, k, M):
                return True
            
            row.pop()
    return False
