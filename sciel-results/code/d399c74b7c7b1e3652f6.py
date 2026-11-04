def _dfs_solve(n_rows, k, max_val, deadline):
    rows = [[0] * (k + 1) for _ in range(n_rows)]
    used_diffs = set()
    nodes_visited = 0
    node_limit = 50000 # Adjust if needed

    def dfs(r_idx, c_idx):
        nonlocal nodes_visited
        if time.time() > deadline:
            return False
        if nodes_visited > node_limit:
            return False
            
        nodes_visited += 1
        
        if r_idx == n_rows:
            return True
            
        if c_idx == k + 1:
            return dfs(r_idx + 1, 0)
            
        if c_idx == 0:
            rows[r_idx][0] = 0
            return dfs(r_idx, 1)
            
        # Symmetry breaking: Order rows by their first element (index 1)
        if c_idx == 1:
            if r_idx > 0:
                start_val = rows[r_idx - 1][1] + 1
            else:
                start_val = 1
        else:
            start_val = rows[r_idx][c_idx - 1] + 1
            
        for v in range(start_val, max_val + 1):
            valid = True
            new_diffs = []
            for j in range(c_idx):
                d = v - rows[r_idx][j]
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
                
            if not valid:
                continue
                
            rows[r_idx][c_idx] = v
            for d in new_diffs:
                used_diffs.add(d)
                
            if dfs(r_idx, c_idx + 1):
                return True
                
            for d in new_diffs:
                used_diffs.remove(d)
                
        return False

    if dfs(0, 0):
        return [list(row) for row in rows]
    else:
        return None
