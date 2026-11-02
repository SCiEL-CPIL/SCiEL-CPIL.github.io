def _build_row(k, scope, used_diffs, start_time, deadline):
    """
    Build a single row of k+1 elements starting at 0, all <= scope,
    such that all internal differences are not in used_diffs.
    Returns the row or None.
    """
    # Backtracking search for the row
    def backtrack(row, pos):
        if time.time() > deadline:
            return None
        
        if pos == k:  # Fixed: pos is count of added elements after 0. We need k more.
            return list(row)
        
        # Try each possible next value
        start_val = row[-1] + 1
        
        for val in range(start_val, scope + 1):
            # Check all differences
            new_diffs = []
            conflict = False
            for a in row:
                d = val - a
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)
            
            if conflict:
                continue
            
            # Add to used_diffs
            for d in new_diffs:
                used_diffs.add(d)
            
            # Recurse
            row.append(val)
            result = backtrack(row, pos + 1)
            row.pop()
            
            # Remove from used_diffs
            for d in new_diffs:
                used_diffs.remove(d)
            
            if result is not None:
                return result
        
        return None
    
    # Start with [0]
    return backtrack([0], 0)
