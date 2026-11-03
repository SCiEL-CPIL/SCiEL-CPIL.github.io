<<<<<<< SEARCH
def construct(n_rows, k):
=======
def _final_scope_reduce(rows, n_rows, k, deadline, rng):
    """Final pass: try to reduce scope by 1 by rebuilding rows that contain the max."""
    if not _is_valid(rows, k):
        return rows
    
    rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    # Try multiple rounds of scope reduction
    for round_num in range(3):
        if time.time() > deadline - 2:
            break
        
        current_scope = _get_scope(rows)
        if current_scope <= 0:
            break
        
        # Find rows that contain the current max
        dc = _build_diff_counts(rows)
        max_val = current_scope
        max_rows = []
        for ri in range(n_rows):
            if max_val in rows[ri]:
                max_rows.append(ri)
        
        if not max_rows:
            break
        
        # Try to rebuild each max-row with a lower max
        improved = False
        for ri in max_rows:
            if time.time() > deadline - 2:
                break
            
            _remove_row_diffs(dc, rows[ri])
            new_row = _row_rebuild(rows[ri], k, dc, rng, max_val - 1)
            
            if new_row is not None and max(new_row) < max_val:
                _add_row_diffs(dc, new_row)
                rows[ri] = new_row
                test_scope = _get_scope(rows)
                
                # Verify validity
                if _is_valid(rows, k):
                    if test_scope < best_scope:
                        best_scope = test_scope
                    improved = True
                    max_val = test_scope
                else:
                    # Revert
                    _remove_row_diffs(dc, new_row)
                    _add_row_diffs(dc, rows[ri])
                    rows[ri] = [list(r) for r in rows[ri]]
                    # Rebuild dc from scratch
                    dc = _build_diff_counts(rows)
            else:
                _add_row_diffs(dc, rows[ri])
        
        if not improved:
            break
    
    return rows

def construct(n_rows, k):
>>>>>>> REPLACE
