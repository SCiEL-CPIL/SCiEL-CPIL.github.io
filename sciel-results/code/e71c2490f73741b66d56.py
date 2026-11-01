import time

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set with n_rows rows and k+1 entries per row.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    # Safety margin: 150s limit, use 145s for search
    deadline = start_time + 145
    
    # Lower bound for scope S
    # Total differences = n_rows * k * (k+1) / 2
    min_scope = int(n_rows * k * (k + 1) / 2)
    
    def backtrack(row_idx, pos_idx, current_row, used_diffs, max_allowed):
        """
        Recursive backtracking to build rows.
        row_idx: index of the current row (0 to n_rows-1)
        pos_idx: index of the current position in the row (0 to k)
        current_row: list of values in the current row so far
        used_diffs: set of all differences used globally
        max_allowed: maximum value allowed for any entry (S)
        """
        # Check time
        if time.time() > deadline:
            return None, False
        
        # Base case: all rows constructed
        if row_idx == n_rows:
            return [], True
        
        # Start of a new row: first element is always 0
        if pos_idx == 0:
            current_row.append(0)
            res, ok = backtrack(row_idx, pos_idx + 1, current_row, used_diffs, max_allowed)
            if ok:
                # Return a copy of the current row and the subsequent rows
                return [list(current_row)] + res, True
            current_row.pop()
            return None, False
            
        last_val = current_row[-1]
        
        # Pruning: if the next value must be > last_val, and last_val + 1 > max_allowed, no solution
        if last_val + 1 > max_allowed:
            return None, False

        # Try next values from last_val + 1 to max_allowed
        for next_val in range(last_val + 1, max_allowed + 1):
            valid = True
            new_diffs = set()
            
            # Check if differences with previous elements in current_row are unused
            for prev_val in current_row:
                diff = next_val - prev_val
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.add(diff)
            
            if not valid:
                continue
                
            # Choose next_val
            current_row.append(next_val)
            used_diffs.update(new_diffs)
            
            # Recurse
            res, ok = backtrack(row_idx, pos_idx + 1, current_row, used_diffs, max_allowed)
            
            if ok:
                # Found a valid completion
                return [list(current_row)] + res, True
            
            # Backtrack
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
        return None, False

    # Search for the minimal scope S
    # We know a solution exists at 134.
    # We search from min_scope up to 140.
    for S in range(min_scope, 141):
        # Check time before starting a new S
        if time.time() > deadline - 5:
            break
            
        res, ok = backtrack(0, 0, [], set(), S)
        if ok:
            # Verify the solution
            all_diffs = set()
            valid = True
            for row in res:
                if len(row) != k + 1:
                    valid = False
                    break
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            if valid:
                return res

    # Fallback: Greedy construction if backtracking fails or times out
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
