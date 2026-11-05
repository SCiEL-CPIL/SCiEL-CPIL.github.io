        if new_val <= 0: continue
        if new_val >= rows[row_idx][pos + 1] ...: continue
        if new_val <= rows[row_idx][pos - 1]: continue
        