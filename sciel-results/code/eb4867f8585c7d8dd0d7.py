import heapq
import time

def construct(n_rows, k):
    # 1. Greedy Baseline for Upper Bound
    def greedy():
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

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    # Precompute binomials for heuristic
    def C2(n):
        return n * (n - 1) // 2

    diffs_per_full_row = C2(k + 1)
    
    start_time = time.time()
    timeout = 115.0
    
    # A* Search
    # State: (f, h, i, row_tuple, used_mask, max_prev)
    # i: current row index
    # row_tuple: current partial row
    # used_mask: bitmask of used differences
    # max_prev: max element in all previous completed rows
    
    def get_heuristic(i, row, used_mask, max_prev):
        # Current max
        curr_max = max(max_prev, row[-1] if row else 0)
        
        # Diffs needed
        diffs_cur_rem = diffs_per_full_row - C2(len(row))
        rows_rem = n_rows - i - 1
        diffs_full_rem = rows_rem * diffs_per_full_row
        total_need = diffs_cur_rem + diffs_full_rem
        
        if total_need == 0:
            return curr_max
        
        # Find the total_need-th smallest positive integer not in used_mask
        found = 0
        d = 1
        while found < total_need:
            if not (used_mask & (1 << d)):
                found += 1
            d += 1
        
        max_diff_needed = d - 1
        return max(curr_max, max_diff_needed)

    # Initial State
    init_row = (0,)
    init_mask = 0
    init_max_prev = 0
    init_h = get_heuristic(0, init_row, init_mask, init_max_prev)
    init_f = max(init_max_prev, init_row[-1], init_h)
    
    counter = 0
    open_list = []
    heapq.heappush(open_list, (init_f, init_h, counter, 0, init_row, init_mask, init_max_prev))
    counter += 1
    
    visited = set()
    parent = {}
    
    while open_list:
        if time.time() - start_time > timeout:
            break
            
        f, h, _, i, row, used_mask, max_prev = heapq.heappop(open_list)
        
        state_key = (i, row, used_mask)
        if state_key in visited:
            continue
        visited.add(state_key)
        
        current_g = max(max_prev, row[-1] if row else 0)
        if current_g >= best_scope:
            continue
            
        if i == n_rows - 1 and len(row) == k + 1:
            if current_g < best_scope:
                best_scope = current_g
                solution_rows = [list(row)]
                curr = state_key
                while curr in parent:
                    p_key, action = parent[curr]
                    p_i, p_row, p_mask = p_key
                    if action[0] == 'next':
                        solution_rows.append(list(p_row))
                    curr = p_key
                solution_rows.reverse()
                best_rows = solution_rows
                return best_rows
        
        if len(row) < k + 1:
            last = row[-1]
            for x in range(last + 1, best_scope + 1):
                new_diffs = [x - a for a in row]
                valid = True
                new_mask = used_mask
                for d in new_diffs:
                    if used_mask & (1 << d):
                        valid = False
                        break
                    new_mask |= (1 << d)
                
                if not valid:
                    continue
                
                new_row = row + (x,)
                new_max_prev = max_prev
                new_g = max(new_max_prev, x)
                
                new_h = get_heuristic(i, new_row, new_mask, new_max_prev)
                new_f = max(new_g, new_h)
                
                if new_f >= best_scope:
                    continue
                
                new_key = (i, new_row, new_mask)
                if new_key not in visited:
                    parent[new_key] = (state_key, ('add', x))
                    heapq.heappush(open_list, (new_f, new_h, counter, i, new_row, new_mask, new_max_prev))
                    counter += 1
                    
        elif len(row) == k + 1 and i < n_rows - 1:
            new_i = i + 1
            new_row = (0,)
            new_mask = used_mask
            new_max_prev = max(max_prev, row[-1])
            
            new_g = new_max_prev
            new_h = get_heuristic(new_i, new_row, new_mask, new_max_prev)
            new_f = max(new_g, new_h)
            
            if new_f >= best_scope:
                continue
            
            new_key = (new_i, new_row, new_mask)
            if new_key not in visited:
                parent[new_key] = (state_key, ('next', None))
                heapq.heappush(open_list, (new_f, new_h, counter, new_i, new_row, new_mask, new_max_prev))
                counter += 1

    return best_rows
