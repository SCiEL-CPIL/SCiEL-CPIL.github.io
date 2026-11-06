import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    max_n = 20
    edges = [(i, j) for i in range(max_n) for j in range(i + 1, max_n)]
    num_edges = len(edges)
    
    # State for current n
    # We will maintain colors for all edges, but only use those for current n
    colors = [-1] * num_edges
    adj_list = [[[] for _ in range(max_n)] for _ in range(4)]
    
    def has_c6(i, j, c):
        # Check if path of length 5 exists from i to j in color c
        # Using the current adj_list
        
        # To speed up, we can use a simple BFS/DFS with depth limit
        # But recursion is okay.
        
        # We can also use a heuristic: 
        # If i and j are not connected in 5 steps, no C6.
        
        # Let's use a non-recursive stack to avoid overhead?
        # Or just recursive.
        
        def dfs(start, end, depth, visited):
            if depth == 0:
                return start == end
            for nb in adj_list[c][start]:
                if nb == end and depth == 1:
                    return True
                if nb not in visited:
                    visited.add(nb)
                    if dfs(nb, end, depth - 1, visited):
                        return True
                    visited.remove(nb)
            return False
            
        return dfs(i, j, 5, {i})

    def backtrack(n, edge_idx):
        if time.time() > deadline:
            return False
            
        for e in range(edge_idx, num_edges):
            i, j = edges[e]
            if i >= n or j >= n:
                continue
                
            # This is the next edge to assign
            for c in range(4):
                # Symmetry breaking
                if n >= 2 and i == 0 and j == 1:
                    if c != 0:
                        continue
                # Additional symmetry breaking for small n
                # If n >= 3 and (i,j) == (0,2):
                #   If colors of (0,1) is 0, we can restrict c(0,2) to {0,1}?
                #   Let's keep it simple for now.
                
                if has_c6(i, j, c):
                    continue
                    
                colors[e] = c
                adj_list[c][i].append(j)
                adj_list[c][j].append(i)
                
                if backtrack(n, e + 1):
                    return True
                    
                colors[e] = -1
                adj_list[c][i].remove(j)
                adj_list[c][j].remove(i)
                
            return False
        return True

    best_n = 1
    best_colors = []
    
    # Start with n=1
    # Try to extend to n+1
    for n in range(2, max_n + 1):
        if time.time() > deadline:
            break
            
        # We need to assign colors for new edges in K_n
        # New edges are those with max(i,j) == n-1
        # But our backtrack function assigns ALL unassigned edges in K_n.
        # Since we are extending from a valid K_{n-1}, the edges of K_{n-1} are already assigned.
        # We just need to assign the new edges.
        
        # However, the `backtrack` function above starts from `edge_idx` and scans.
        # It will skip already assigned edges?
        # No, it checks `if i >= n or j >= n: continue`.
        # It does NOT check if `colors[e] != -1`.
        # So it will re-assign edges of K_{n-1}.
        
        # We should modify backtrack to skip assigned edges.
        
        def backtrack_ext(n, edge_idx):
            if time.time() > deadline:
                return False
                
            for e in range(edge_idx, num_edges):
                i, j = edges[e]
                if i >= n or j >= n:
                    continue
                if colors[e] != -1:
                    continue
                    
                for c in range(4):
                    if n >= 2 and i == 0 and j == 1:
                        if c != 0:
                            continue
                            
                    if has_c6(i, j, c):
                        continue
                        
                    colors[e] = c
                    adj_list[c][i].append(j)
                    adj_list[c][j].append(i)
                    
                    if backtrack_ext(n, e + 1):
                        return True
                        
                    colors[e] = -1
                    adj_list[c][i].remove(j)
                    adj_list[c][j].remove(i)
                    
                return False
            return True
            
        if backtrack_ext(n, 0):
            best_n = n
            # Extract colors
            valid_colors = []
            for e, (i, j) in enumerate(edges):
                if i < n and j < n:
                    valid_colors.append(colors[e])
            best_colors = valid_colors
        else:
            break
            
    return {'n': best_n, 'colors': best_colors}
