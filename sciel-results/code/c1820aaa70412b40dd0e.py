    def apply_move(r, idx, new_val):
        nonlocal rows, used, current_scope
        
        old_val = rows[r][idx]
        if new_val == old_val:
            return current_scope
        
        # Check increasing order
        if idx > 0 and new_val <= rows[r][idx - 1]:
            return None # Invalid
        if idx < k and new_val >= rows[r][idx + 1]:
            return None # Invalid
        
        old_diffs = get_diffs(rows[r], idx)
        
        # Check if new diffs conflict
        new_diffs = set()
        for i in range(idx):
            new_diffs.add(new_val - rows[r][i])
        for i in range(idx + 1, k + 1):
            new_diffs.add(rows[r][i] - new_val)
            
        conflict = new_diffs & (used - old_diffs)
        if conflict:
            return None # Invalid
        
        # Calculate new scope
        new_scope = current_scope
        # Determine if scope changes
        # If new_val > current_scope, scope increases
        # If old_val == current_scope and new_val < old_val, scope might decrease
        # Else scope stays same
        
        # To be safe and simple, just update row, calc max, and revert if needed?
        # Cost of max is O(N*K). N=4, K=8. Very cheap.
        
        # Save old state
        old_row_val = rows[r][idx]
        
        # Tentative update
        rows[r][idx] = new_val
        tentative_scope = max(max(row) for row in rows)
        
        delta = tentative_scope - current_scope
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            # Accept
            used -= old_diffs
            used |= new_diffs
            current_scope = tentative_scope
            # rows is already updated
            return current_scope
        else:
            # Reject
            rows[r][idx] = old_row_val
            return current_scope
