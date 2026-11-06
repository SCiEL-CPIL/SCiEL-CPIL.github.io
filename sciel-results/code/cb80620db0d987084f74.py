import time

def construct():
    # Try to find a valid coloring for n=20, 21, 22...
    # Backtracking is feasible for small n
    for n in [20, 21, 22, 23, 24, 25, 26]:
        if time.time() > 115:
            break
        colors = try_backtracking(n)
        if colors is not None:
            return {'n': n, 'colors': colors}
    
    # Fallback to a known small valid coloring or greedy result
    # Greedy for n=11
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def try_backtracking(n):
    # edges in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    # color[u][v] = c or -1
    color = [[-1]*n for _ in range(n)]
    # neighbors[c][u] = set of v such that color[u][v] == c
    neighbors = [ [set() for _ in range(n)] for _ in range(5) ]
    
    def is_valid(c, u, v):
        # Check if adding (u,v) with color c creates a C4
        # 1. |N_c(u) intersect N_c(v)| <= 1
        inter = neighbors[c][u].intersection(neighbors[c][v])
        if len(inter) > 1:
            return False
        
        # 2. No w in N_c(v) such that N_c(u) intersect N_c(w) is non-empty
        #    This means u and w cannot have a common neighbor in c, 
        #    otherwise u-v-w-x-u is a C4?
        #    Wait, the condition "No two vertices with two common neighbours" is equivalent to no C4.
        #    Adding (u,v) creates a C4 if:
        #    a) There exists x, y such that u-x, x-v, v-y, y-u are in C.
        #       This implies x is in N_c(u) \cap N_c(v). If |intersection| >= 2, say x,y, 
        #       then u-x-v-y-u is a C4? No, u-x and x-v are edges. v-y and y-u are edges.
        #       Yes, x and y are common neighbors of u and v.
        #       So condition 1 covers this.
        #    b) There exists x, y such that u-x, x-y, y-v, v-u are in C?
        #       This is a C4 u-x-y-v-u.
        #       Here x is in N_c(u). y is in N_c(v).
        #       And x-y is an edge in C.
        #       This means x and y are connected.
        #       So we need to check if there is an edge between N_c(u) and N_c(v) in color c?
        #       No, x-y is an edge in color c.
        #       So we need to check if any x in N_c(u) is connected to any y in N_c(v) in color c.
        #       Actually, if x in N_c(u) and y in N_c(v), and (x,y) is in C, then u-x-y-v-u is a C4.
        #       So we need N_c(u) and N_c(v) to be "independent" in the graph G_c?
        #       i.e., No edge in G_c connects a vertex in N_c(u) to a vertex in N_c(v).
        
        for x in neighbors[c][u]:
            for y in neighbors[c][v]:
                if x == y: continue # This is case (a), already handled by intersection size
                # Check if (x,y) is in color c
                if y in neighbors[c][x]:
                    return False
        return True

    def backtrack(idx):
        if idx == len(edges):
            return True
            
        u, v = edges[idx]
        for c in range(5):
            if is_valid(c, u, v):
                # Assign
                color[u][v] = c
                color[v][u] = c
                neighbors[c][u].add(v)
                neighbors[c][v].add(u)
                
                if backtrack(idx + 1):
                    return True
                    
                # Unassign
                color[u][v] = -1
                color[v][u] = -1
                neighbors[c][u].remove(v)
                neighbors[c][v].remove(u)
                
        return False

    # To help backtracking, we can shuffle edges? 
    # Lexicographic is fine for small n.
    if backtrack(0):
        out_colors = []
        for i in range(n):
            for j in range(i+1, n):
                out_colors.append(color[i][j])
        return out_colors
    return None
