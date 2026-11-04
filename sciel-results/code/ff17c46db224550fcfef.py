def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0  # 5s margin from 120s limit
    
    def deterministic_greedy():
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

    best_rows = deterministic_greedy()
    best_scope = max(max(r) for r in best_rows)
    
    def backtrack_search(max_scope, seed):
        random.seed(seed)
        used_diffs = set()
        rows = []
        
        def dfs(row_idx, current_row):
            nonlocal used_diffs, rows
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                if max(current_row) > max_scope:
                    return False
                
                rows.append(current_row)
                if dfs(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Pruning: minimum possible max value for this row
            min_possible_max = last_val + (k + 1 - len(current_row))
            if min_possible_max > max_scope:
                return False

            candidates = []
            for x in range(last_val + 1, max_scope + 1):
                new_diffs = {x - a for a in current_row}
                if not (new_diffs & used_diffs):
                    candidates.append((x, new_diffs))
            
            if not candidates:
                return False
            
            # Shuffle to allow different paths with different seeds
            random.shuffle(candidates)
            
            for x, new_diffs in candidates:
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if dfs(row_idx, current_row):
                    return True
                
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        if dfs(0, [0]):
            scope = max(max(r) for r in rows)
            return rows, scope
        return None, float('inf')

    # Search loop
    current_target = best_scope - 1
    seed = 0
    
    while time.time() - start_time < deadline:
        if current_target < k:
            break
            
        rows, scope = backtrack_search(current_target, seed)
        
        if rows is not None:
            best_scope = scope
            best_rows = rows
            current_target = best_scope - 1
            seed += 1
        else:
            # No solution exists for this scope (DFS is complete)
            break

    return best_rows
