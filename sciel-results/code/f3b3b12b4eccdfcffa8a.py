import time
import numpy as np

def construct():
    start_time = time.time()
    
    # We will try to find a valid coloring for n=27, then 26, etc.
    # The known bound is likely 26 or 27.
    # We use a greedy backtracking algorithm with bitset optimization.
    
    def try_build(n, max_time=110):
        # adj[c] is a list of bitsets (integers) for each vertex.
        # adj[c][v] has bit u set if edge (u,v) is color c.
        adj = [ [0] * n for _ in range(5) ]
        # colors[i][j] stores the color of edge (i,j)
        colors = [[0] * n for _ in range(n)]
        
        # To speed up intersection checks, we can precompute 
        # "conflict" info? 
        # The check for adding edge (u, v) with color c is:
        # For all w < u such that color(w, v) == c, we need (adj[c][u] & adj[c][w]) == 0.
        # This is O(v) per check, which is O(n^2) per vertex, O(n^3) total.
        # For n=27, n^3 is ~20k, which is fast.
        
        def backtrack(v):
            if time.time() - start_time > max_time:
                return False
            
            if v == n:
                return True
            
            # Heuristic: Order the edges (u, v) by the number of available colors?
            # Or just assign u=0 to v-1.
            # Let's try to assign colors to u=0, 1, ..., v-1.
            
            # We can use a recursive helper for the edges of v
            def assign_edges(idx):
                if idx == v:
                    return backtrack(v + 1)
                
                u = idx
                # Try colors 0..4
                # Heuristic: Try colors that are "popular" or "least constrained"?
                # For now, try 0..4
                
                for c in range(5):
                    # Check validity:
                    # If we assign color c to (u, v), then for any w < u 
                    # such that colors[w][v] == c, we must have 
                    # adj[c][u] & adj[c][w] == 0.
                    
                    valid = True
                    # We need to check against all w < u that have already been assigned color c for edge (w, v)
                    # Note: colors[w][v] is only set for w < idx if we've processed it.
                    # In this loop, we are processing u = idx.
                    # Previous u's are 0...idx-1.
                    for w in range(idx):
                        if colors[w][v] == c:
                            if adj[c][u] & adj[c][w]:
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    # Assign
                    colors[u][v] = c
                    colors[v][u] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if assign_edges(idx + 1):
                        return True
                    
                    # Unassign
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
                    colors[u][v] = 0
                    colors[v][u] = 0
                
                return False
            
            return assign_edges(0)
        
        # Initial state: Vertex 0 is isolated.
        # Start backtracking from v=1.
        # We can add symmetry breaking:
        # Edge (0, 1) must be color 0.
        colors[0][1] = 0
        colors[1][0] = 0
        adj[0][0] |= (1 << 1)
        adj[0][1] |= (1 << 0)
        
        if n > 1:
            if backtrack(2):
                return True, adj, colors
        else:
            return True, adj, colors
            
        return False, adj, colors

    for n in range(27, 10, -1):
        if time.time() - start_time > 115:
            break
            
        found, adj, colors = try_build(n)
        if found:
            color_list = []
            for i in range(n):
                for j in range(i+1, n):
                    color_list.append(colors[i][j])
            return {'n': n, 'colors': color_list}
            
    # Fallback
    return {'n': 2, 'colors': [0]}
