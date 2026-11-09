import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Fallback to seed solution
    n_seed = 9
    best_n = n_seed
    best_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    
    # Try to find solutions for larger n
    for n in range(20, 8, -1):
        if time.time() - start_time > deadline:
            break
            
        solution = _solve_n(n, start_time, deadline)
        if solution is not None:
            return {'n': n, 'colors': solution}
            
    return {'n': best_n, 'colors': best_colors}

def _solve_n(n, start_time, deadline):
    total_edges = n * (n - 1) // 2
    colors = [-1] * total_edges
    
    # adj_bits[c][u] : bitmask of neighbors of u in color c
    adj_bits = [ [0] * n for _ in range(4) ]
    
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Symmetry breaking: Fix colors of first few edges
    forced = {}
    if n >= 4:
        forced[(0,1)] = 0
        forced[(0,2)] = 1
        forced[(0,3)] = 2
        
    def get_edge(i, j):
        if i > j: i, j = j, i
        return i * n - i*(i+1)//2 + j - i - 1

    def check_c6(u, v, c):
        visited_mask = 1 << u
        
        def dfs(node, depth, mask):
            if depth == 5:
                return node == v
            
            neighbors = adj_bits[c][node]
            while neighbors:
                lsb = neighbors & -neighbors
                nei = lsb.bit_length() - 1
                neighbors ^= lsb
                
                if (mask >> nei) & 1:
                    continue
                    
                if dfs(nei, depth + 1, mask | lsb):
                    return True
            return False
            
        return dfs(u, 0, visited_mask)

    def backtrack(k):
        if time.time() - start_time > deadline:
            return False
            
        if k == total_edges:
            return True
            
        i, j = edges[k]
        e_idx = get_edge(i, j)
        
        if (i, j) in forced:
            c = forced[(i, j)]
            if not check_c6(i, j, c):
                colors[e_idx] = c
                adj_bits[c][i] |= (1 << j)
                adj_bits[c][j] |= (1 << i)
                
                if backtrack(k + 1):
                    return True
                    
                adj_bits[c][i] &= ~(1 << j)
                adj_bits[c][j] &= ~(1 << i)
                colors[e_idx] = -1
            return False
            
        for c in range(4):
            if check_c6(i, j, c):
                continue
                
            colors[e_idx] = c
            adj_bits[c][i] |= (1 << j)
            adj_bits[c][j] |= (1 << i)
            
            if backtrack(k + 1):
                return True
                
            adj_bits[c][i] &= ~(1 << j)
            adj_bits[c][j] &= ~(1 << i)
            colors[e_idx] = -1
            
        return False

    if backtrack(0):
        return colors
        
    return None
