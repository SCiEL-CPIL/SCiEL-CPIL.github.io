import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
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
    best_max = max(max(row) for row in best_rows)
    
    used_diffs = set()
    current_rows = []
    
    def dfs(row_idx):
        nonlocal best_rows, best_max
        
        if row_idx == n_rows:
            current_max = max(max(row) for row in current_rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = [row[:] for row in current_rows]
            return
        
        if time.time() > deadline:
            return
        
        # Symmetry breaking: The max of the current row must be >= max of previous row
        min_max_allowed = current_rows[row_idx - 1][-1] if row_idx > 0 else 0
        
        row = [0]
        
        def build_row(current_row, last_val):
            nonlocal used_diffs
            
            if len(current_row) == k + 1:
                row_max = current_row[-1]
                
                # Symmetry break check
                if row_max < min_max_allowed:
                    return
                
                # Check if this row's diffs conflict with global used_diffs
                row_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        row_diffs.add(current_row[j] - current_row[i])
                
                if row_diffs & used_diffs:
                    return
                
                # Add to global state
                used_diffs.update(row_diffs)
                current_rows.append(current_row)
                
                dfs(row_idx + 1)
                
                # Backtrack
                current_rows.pop()
                for d in row_diffs:
                    used_diffs.discard(d)
                
                return
            
            # Pruning: If the last element is already >= best_max, we can't improve
            # Note: We are looking for max < best_max.
            # So if current_row[-1] >= best_max, we prune.
            if current_row[-1] >= best_max:
                return
            
            # Also, if the last element is < min_max_allowed, we might still be able to reach it?
            # No, the max is the last element. So if last element < min_max_allowed, 
            # we can still add larger elements. 
            # But if last element + (remaining elements needed) * (min gap) > best_max?
            # Simple bound: last_val < best_max.
            
            for x in range(last_val + 1, best_max): 
                # Check conflict with current row and global
                conflict = False
                new_diffs = set()
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                build_row(current_row, x)
                
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
                
                # Time check inside loop
                if time.time() > deadline:
                    return

        build_row(row, 0)

    dfs(0)
    
    return best_rows
