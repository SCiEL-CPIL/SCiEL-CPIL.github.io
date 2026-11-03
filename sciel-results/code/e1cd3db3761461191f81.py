def _compress(rows, k, deadline):
    counts = _get_diff_counts(rows)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(rows)):
            for j in range(k, 0, -1):
                old_val = rows[i][j]
                new_val = old_val - 1
                if new_val <= rows[i][j-1]:
                    continue
                
                # Try move
                _remove_row(counts, rows[i])
                rows[i][j] = new_val
                
                valid_local = True
                if new_val <= rows[i][j-1]: valid_local = False
                if j < k and new_val >= rows[i][j+1]: valid_local = False
                
                if valid_local:
                    _add_row(counts, rows[i])
                    if _get_conflicts(counts) == 0:
                        improved = True
                        break
                    else:
                        # Revert
                        _remove_row(counts, rows[i])
                        rows[i][j] = old_val
                        _add_row(counts, rows[i])
                else:
                    # Revert
                    rows[i][j] = old_val
                    _add_row(counts, rows[i])
                        
    return rows
