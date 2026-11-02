# EVOLVE-BLOCK-START
import time
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    """
    # Lower bound estimation
    # Total distinct differences required: n_rows * k * (k+1) / 2
    # Max differences in [0, M] is M * (M+1) / 2
    target_diffs = n_rows * k * (k + 1) // 2
    m_lb = 1
    while m_lb * (m_lb + 1) // 2 < target_diffs:
        m_lb += 1
    
    # Upper bound heuristic
    # Greedy gives a decent upper bound
    m_ub = _greedy_upper_bound(n_rows, k)
    
    # Binary search for the minimum M
    lo = m_lb
    hi = m_ub
    
    best_rows = None
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0
    
    # If the range is small, just try from lo upwards
    # If range is large, binary search might miss the exact min if we don't store best, 
    # but we want the MINIMUM. Binary search finds the smallest M for which a solution exists.
    
    while lo <= hi:
        if time.time() - start_time > time_limit:
            break
            
        mid = (lo + hi) // 2
        
        # Check if a solution exists with max value <= mid
        rows = _backtrack_solve(n_rows, k, mid)
        
        if rows is not None:
            best_rows = rows
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_rows is None:
        # Fallback if binary search didn't find anything (e.g. timeout or failure)
        # Return the greedy solution
        best_rows = _greedy_construct(n_rows, k)
        
    return best_rows

def _backtrack_solve(n_rows, k, max_val):
    """
    Attempt to construct n_rows rows with values in [0, max_val] using backtracking.
    Returns list of rows if successful, None otherwise.
    """
    used_diffs = set()
    rows = []
    
    # Symmetry breaking: 
    # Rows are indistinguishable. We can enforce an ordering on the first non-zero element.
    # Let x_i[1] be the first non-zero element of row i.
    # We enforce x_0[1] <= x_1[1] <= ... <= x_{n-1}[1].
    
    def backtrack(row_idx, current_row, last_val, min_first_val):
        """
        row_idx: which row we are currently constructing
        current_row: list of values in current row so far
        last_val: last value added to current_row
        min_first_val: minimum allowed value for current_row[1] (symmetry breaking)
        """
        if row_idx == n_rows:
            return True
            
        # If we are starting a new row
        if len(current_row) == 0:
            # We need to choose x[1] >= min_first_val
            # And x[k] <= max_val
            # Heuristic: Try smaller x[1] first? Or larger?
            # Smaller x[1] leaves more room for larger values, but uses smaller diffs.
            # Let's try values from min_first_val upwards.
            
            # Optimization: If we have used many diffs, we might need larger values?
            # Let's just iterate.
            
            for x1 in range(min_first_val, max_val + 1):
                # Pruning: Can we fit k-1 more elements in [x1+1, max_val]?
                # We need at least k-1 distinct integers greater than x1 up to max_val.
                # So max_val - x1 >= k - 1  => x1 <= max_val - (k - 1)
                if x1 > max_val - (k - 1):
                    break
                
                # Check if diff x1 - 0 = x1 is available
                if x1 in used_diffs:
                    continue
                
                # Tentatively add x1
                # New diffs: {x1}
                # Check if x1 conflicts with used_diffs (already checked)
                
                # Recurse to next element in this row
                # We pass a copy of used_diffs? No, we modify and backtrack.
                
                # Store previous state
                # Since we are adding one element at a time, we can manage the set carefully.
                
                # Let's refactor to pass the set by reference and undo changes.
                
                # Add x1 to row
                current_row.append(x1)
                # Add diff x1 to used_diffs
                used_diffs.add(x1)
                
                # Recurse to fill rest of the row
                if backtrack(row_idx, current_row, x1, min_first_val):
                    return True
                
                # Backtrack
                used_diffs.remove(x1)
                current_row.pop()
                
            return False
        
        else:
            # We are continuing a row
            # current_row has length >= 1
            # We need to add next element x > last_val
            # x <= max_val
            
            # Determine how many more elements we need
            # len(current_row) is current count. We need k+1 total.
            # So we need (k + 1 - len(current_row)) more elements.
            
            remaining_needed = k + 1 - len(current_row)
            
            # Pruning: Can we fit remaining_needed elements in [last_val+1, max_val]?
            if max_val - last_val < remaining_needed:
                return False
            
            # Iterate possible next values
            # Start from last_val + 1
            # To reduce symmetry or improve pruning, we might want to be smart here.
            # But simple iteration works.
            
            for x in range(last_val + 1, max_val + 1):
                # Check if we can fit the rest after x
                # We need remaining_needed - 1 more elements after x.
                # So max_val - x >= remaining_needed - 1
                if x > max_val - (remaining_needed - 1):
                    break
                
                # Check differences
                # New diffs: {x - y for y in current_row}
                # These must be distinct from each other (guaranteed since y < x and y distinct? 
                # Wait, x-y for different y are distinct if y are distinct. Yes.)
                # They must be distinct from used_diffs.
                
                valid = True
                new_diffs = []
                for y in current_row:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Check if new_diffs have duplicates among themselves?
                # No, because y are distinct, so x-y are distinct.
                
                # Commit
                current_row.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                
                # Recurse
                if backtrack(row_idx, current_row, x, min_first_val):
                    return True
                
                # Backtrack
                for d in new_diffs:
                    used_diffs.remove(d)
                current_row.pop()
                
            return False

    # Start backtracking
    # We need to store the final rows. 
    # The backtrack function above modifies `rows`? No, it doesn't.
    # We need to capture the solution.
    
    # Let's modify backtrack to return the solution or store it in a closure variable.
    
    solution = [None]
    
    def backtrack_with_store(row_idx, current_row, last_val, min_first_val):
        if row_idx == n_rows:
            solution[0] = [r[:] for r in rows]
            return True
            
        if len(current_row) == 0:
            # Starting a new row
            # Symmetry: x1 >= min_first_val
            for x1 in range(min_first_val, max_val + 1):
                if x1 > max_val - (k - 1):
                    break
                if x1 in used_diffs:
                    continue
                
                current_row.append(x1)
                used_diffs.add(x1)
                
                if backtrack_with_store(row_idx, current_row, x1, min_first_val):
                    return True
                
                used_diffs.remove(x1)
                current_row.pop()
            return False
        else:
            remaining_needed = k + 1 - len(current_row)
            if max_val - last_val < remaining_needed:
                return False
                
            for x in range(last_val + 1, max_val + 1):
                if x > max_val - (remaining_needed - 1):
                    break
                
                valid = True
                new_diffs = []
                for y in current_row:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                current_row.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                
                if backtrack_with_store(row_idx, current_row, x, min_first_val):
                    return True
                
                for d in new_diffs:
                    used_diffs.remove(d)
                current_row.pop()
            return False

    # Initial call
    # We need to handle the storage of completed rows.
    # The current `backtrack_with_store` doesn't add to `rows` when a row is complete.
    # We need to add the row to `rows` when we finish a row.
    
    # Let's adjust the logic:
    # When we finish a row (len(current_row) == k+1), we add it to `rows`, 
    # clear `current_row`, and move to next row_idx.
    
    # Actually, the recursion above mixes row completion and element addition.
    # Let's restructure:
    
    def solve():
        used_diffs = set()
        rows = []
        
        def bt(row_idx, current_row, last_val, min_first_val):
            if row_idx == n_rows:
                return True
            
            if len(current_row) == 0:
                # Start new row
                for x1 in range(min_first_val, max_val + 1):
                    if x1 > max_val - (k - 1):
                        break
                    if x1 in used_diffs:
                        continue
                    
                    current_row.append(x1)
                    used_diffs.add(x1)
                    
                    if bt(row_idx, current_row, x1, min_first_val):
                        return True
                    
                    used_diffs.remove(x1)
                    current_row.pop()
                return False
            else:
                # Continue row
                if len(current_row) == k + 1:
                    # Row complete
                    rows.append(current_row[:])
                    # Next row
                    # Symmetry: next row's first element must be >= current row's first element
                    next_min_first = current_row[1]
                    return bt(row_idx + 1, [], 0, next_min_first)
                
                remaining_needed = k + 1 - len(current_row)
                if max_val - last_val < remaining_needed:
                    return False
                
                for x in range(last_val + 1, max_val + 1):
                    if x > max_val - (remaining_needed - 1):
                        break
                    
                    valid = True
                    new_diffs = []
                    for y in current_row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    current_row.append(x)
                    for d in new_diffs:
                        used_diffs.add(d)
                    
                    if bt(row_idx, current_row, x, min_first_val):
                        return True
                    
                    for d in new_diffs:
                        used_diffs.remove(d)
                    current_row.pop()
                return False
        
        if bt(0, [], 0, 0):
            return rows
        return None

    return solve()

def _greedy_upper_bound(n_rows, k):
    """Simple greedy to get an upper bound."""
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
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k):
    """Fallback greedy construction."""
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

# EVOLVE-BLOCK-END
