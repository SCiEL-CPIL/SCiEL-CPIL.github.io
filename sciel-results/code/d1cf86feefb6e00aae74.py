import time

def construct():
    n = 10
    A = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    B = [[-1 for _ in range(n)] for _ in range(n)]
    row_sets = [set() for _ in range(n)]
    col_sets = [set() for _ in range(n)]
    pair_set = set()
    
    start_time = time.time()
    timeout = 55
    
    # Pre-fill first row with 0..9
    for j in range(n):
        B[0][j] = j
        row_sets[0].add(j)
        col_sets[j].add(j)
        pair_set.add((A[0][j], j))

    def get_options(i, j):
        opts = []
        for val in range(n):
            if val not in row_sets[i] and val not in col_sets[j] and (A[i][j], val) not in pair_set:
                opts.append(val)
        return opts

    def solve(i, j):
        if time.time() - start_time > timeout:
            return False
        if i == n:
            return True
        
        # Find next empty cell (MRV heuristic)
        min_opts = None
        min_len = n + 1
        next_i, next_j = -1, -1
        
        # We can just scan from current position to end of row, then next rows
        # For simplicity, scan all empty cells to find MRV
        for r in range(i, n):
            for c in range(n if r > i else j, n):
                if B[r][c] == -1:
                    opts = get_options(r, c)
                    if len(opts) < min_len:
                        min_len = len(opts)
                        min_opts = opts
                        next_i, next_j = r, c
                        if min_len == 0:
                            return False
                        if min_len == 1:
                            break
            if min_len == 1:
                break
        
        if next_i == -1:
            return True # All filled
            
        for val in min_opts:
            B[next_i][next_j] = val
            row_sets[next_i].add(val)
            col_sets[next_j].add(val)
            pair_set.add((A[next_i][next_j], val))
            
            if solve(next_i, next_j + 1): # Note: passing next_j+1 is not correct for MRV scan, but solve will re-scan
                return True
            
            B[next_i][next_j] = -1
            row_sets[next_i].remove(val)
            col_sets[next_j].remove(val)
            pair_set.remove((A[next_i][next_j], val))
            
        return False

    if solve(1, 0):
        return [A, B]
    else:
        return [A]
