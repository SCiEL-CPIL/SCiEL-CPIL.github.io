import time
import random

def construct(n_rows, k):
    start_time = time.time()
    lower_bound = 112
    
    # Fallback greedy
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
    best_result = [list(r) for r in rows]
    best_scope = max(max(r) for r in best_result)
    
    for M in range(lower_bound, best_scope + 1):
        if time.time() - start_time > 140:
            break
            
        rulers = []
        
        def gen_rulers_rec(current_row, used_mask, last_val):
            if len(current_row) == k + 1:
                rulers.append((tuple(current_row), used_mask))
                return
            
            remaining_slots = (k + 1) - len(current_row)
            if last_val + remaining_slots - 1 > M:
                return
                
            start = last_val + 1
            for next_val in range(start, M + 1):
                new_mask = 0
                conflict = False
                for a in current_row:
                    d = next_val - a
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                    new_mask |= (1 << d)
                
                if conflict:
                    continue
                
                current_row.append(next_val)
                gen_rulers_rec(current_row, used_mask | new_mask, next_val)
                current_row.pop()
                
                if len(rulers) > 5000:
                    return # Limit generation to avoid memory/time issues
                
        gen_rulers_rec([0], 0, 0)
        
        if len(rulers) < n_rows:
            continue
            
        random.shuffle(rulers)
        
        found = False
        selected = []
        nodes_visited = 0
        max_nodes = 100000 # Limit search depth/breadth
        
        def search_compatible(idx, used_mask):
            nonlocal nodes_visited
            if nodes_visited > max_nodes:
                return False
            
            if len(selected) == n_rows:
                return True
            
            for i in range(idx, len(rulers)):
                nodes_visited += 1
                mask = rulers[i][1]
                if mask & used_mask:
                    continue
                
                selected.append(rulers[i])
                if search_compatible(i + 1, used_mask | mask):
                    return True
                selected.pop()
            return False
        
        if search_compatible(0, 0):
            result_rows = [list(r[0]) for r in selected]
            scope = max(max(r) for r in result_rows)
            if scope < best_scope:
                best_scope = scope
                best_result = result_rows
            break

    return best_result
