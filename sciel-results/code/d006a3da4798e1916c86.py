def _find_orthogonal_mate_v2(square1, n, time_limit=50.0):
    start_time = time.time()
    n_restarts = 0
    max_restarts = 100
    
    while time.time() < start_time + time_limit and n_restarts < max_restarts:
        n_restarts += 1
        # Reset state
        s2 = [[None for _ in range(n)] for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_used = set()
        
        # Randomize the order of cells? No, row-major is fine.
        # Randomize values.
        
        def is_valid(r, c, val):
            if val in row_used[r]:
                return False
            if val in col_used[c]:
                return False
            p = (square1[r][c], val)
            if p in pair_used:
                return False
            return True

        def backtrack(r, c):
            if time.time() > start_time + time_limit:
                return False
            if r == n:
                return True
            if c == n:
                return backtrack(r + 1, 0)
            
            vals = list(range(n))
            random.shuffle(vals)
            
            for val in vals:
                if is_valid(r, c, val):
                    s2[r][c] = val
                    row_used[r].add(val)
                    col_used[c].add(val)
                    pair_used.add((square1[r][c], val))
                    
                    if backtrack(r, c + 1):
                        return True
                    
                    s2[r][c] = None
                    row_used[r].remove(val)
                    col_used[c].remove(val)
                    pair_used.remove((square1[r][c], val))
            
            return False

        if backtrack(0, 0):
            return s2
            
    return None
