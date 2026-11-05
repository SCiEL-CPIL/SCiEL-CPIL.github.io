import random
import time

def construct(n_rows, k):
    def greedy_row(used, k):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used.update(new)
        return row

    used = set()
    rows = []
    for _ in range(n_rows):
        r = greedy_row(used, k)
        rows.append(r)
    
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def build_row(forbidden, k, max_val_limit, time_limit=0.1):
        start = time.time()
        row = [0]
        used_in_row = set()
        
        def backtrack(idx, last_val):
            if time.time() - start > time_limit:
                return False
            if idx == k + 1:
                return True
            
            for x in range(last_val + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & forbidden or new_diffs & used_in_row:
                    continue
                row.append(x)
                used_in_row.update(new_diffs)
                if backtrack(idx + 1, x):
                    return True
                row.pop()
                used_in_row.difference_update(new_diffs)
            return False
        
        if backtrack(1, 0):
            return row
        return None

    start_time = time.time()
    timeout = 115.0
    
    while time.time() - start_time < timeout:
        max_idx = 0
        max_val = 0
        for i, r in enumerate(rows):
            if r[-1] > max_val:
                max_val = r[-1]
                max_idx = i
        
        old_diffs = get_diffs(rows[max_idx])
        used.difference_update(old_diffs)
        old_row = rows[max_idx]
        old_max = old_row[-1]
        
        # 1. Try Greedy
        new_row = greedy_row(used, k)
        if new_row[-1] < old_max:
            rows[max_idx] = new_row
            used.update(get_diffs(new_row))
            continue # Restart loop with new state
        
        # 2. Try Backtracking with limit old_max - 1
        new_row = build_row(used, k, old_max - 1, time_limit=0.2)
        if new_row:
            rows[max_idx] = new_row
            used.update(get_diffs(new_row))
            continue
        
        # Revert
        rows[max_idx] = old_row
        used.update(old_diffs)
        
        # Kick: Modify a random other row
        r_idx = random.randint(0, n_rows - 1)
        if r_idx != max_idx:
            old_diffs_r = get_diffs(rows[r_idx])
            used.difference_update(old_diffs_r)
            old_row_r = rows[r_idx]
            old_max_r = old_row_r[-1]
            
            # Try Greedy for this row
            new_row_r = greedy_row(used, k)
            if new_row_r[-1] < old_max_r:
                rows[r_idx] = new_row_r
                used.update(get_diffs(new_row_r))
            else:
                # Try Backtracking
                new_row_r = build_row(used, k, old_max_r - 1, time_limit=0.2)
                if new_row_r:
                    rows[r_idx] = new_row_r
                    used.update(get_diffs(new_row_r))
                else:
                    rows[r_idx] = old_row_r
                    used.update(old_diffs_r)

    return rows
