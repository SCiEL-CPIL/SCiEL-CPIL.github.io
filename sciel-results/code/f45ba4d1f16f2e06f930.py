import time

def construct(n_rows, k):
    def greedy_construct():
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

    best_rows = greedy_construct()
    best_scope = max(r[-1] for r in best_rows)
    
    start_time = time.time()
    deadline = start_time + 110
    
    search_state = {
        'best_scope': best_scope,
        'best_rows': best_rows,
        'deadline': deadline
    }
    
    total_diffs_per_row = k * (k + 1) // 2
    
    def popcount(x):
        try:
            return x.bit_count()
        except AttributeError:
            return bin(x).count('1')

    def backtrack(rows, used_mask, current_row, row_mask, current_max):
        if time.time() > search_state['deadline']:
            return
            
        if len(rows) == n_rows:
            scope = current_row[-1]
            if scope < search_state['best_scope']:
                search_state['best_scope'] = scope
                search_state['best_rows'] = rows + [current_row]
            return
        
        if len(current_row) == k + 1:
            total_used = popcount(used_mask) + popcount(row_mask)
            rows_left = n_rows - len(rows) - 1
            diffs_needed = rows_left * total_diffs_per_row
            max_possible_diff = search_state['best_scope'] - 1
            available = max_possible_diff - total_used
            if available < diffs_needed:
                return
            
            backtrack(rows + [current_row], used_mask | row_mask, [0], 0, 0)
            return
        
        last = current_row[-1]
        for x in range(last + 1, search_state['best_scope']):
            new_diffs_mask = 0
            for a in current_row:
                d = x - a
                new_diffs_mask |= (1 << d)
            
            if (new_diffs_mask & used_mask) or (new_diffs_mask & row_mask):
                continue
                
            new_total_used = popcount(used_mask) + popcount(row_mask) + popcount(new_diffs_mask)
            
            s = len(current_row)
            r = k - s
            diffs_to_complete_row = r * s + r * (r + 1) // 2
            
            rows_left = n_rows - len(rows) - 1
            diffs_needed_future = rows_left * total_diffs_per_row
            
            total_needed = diffs_to_complete_row + diffs_needed_future
            
            max_possible_diff = search_state['best_scope'] - 1
            available = max_possible_diff - new_total_used
            
            if available < total_needed:
                continue
                
            current_row.append(x)
            backtrack(rows, used_mask, current_row, row_mask | new_diffs_mask, x)
            current_row.pop()
            
            if time.time() > search_state['deadline']:
                return

    backtrack([], 0, [0], 0, 0)
    
    return search_state['best_rows']
