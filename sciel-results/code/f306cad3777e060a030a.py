def is_valid_lift_fast(signs):
    # signs: +1/-1
    # voltages: 0 for +1, 1 for -1
    voltages = np.where(signs == 1, 0, 1)
    
    # Build spanning tree and prefix sums
    # n is small, simple DFS/BFS is fine
    n = graph['n']
    visited = [False] * n
    prefix = [0] * n
    parent = [-1] * n
    non_tree_edges = []
    
    # Map edge index to (u, v)
    # We need to traverse.
    # Since n is small (<=18), we can just do a simple DFS.
    
    stack = [0]
    visited[0] = True
    prefix[0] = 0
    
    # We need to know which edges are tree edges.
    # Let's just iterate edges and add to tree if it connects new node.
    # This is a standard Kruskal-like or just DFS on the graph.
    
    # Actually, simpler: just check if the system $x_u + x_v = vol(u,v)$ is consistent.
    # If consistent, disconnected. If inconsistent, connected.
    # This is a 2-SAT / bipartite coloring check on the signed graph.
    # Color vertices 0/1.
    # Constraint: color[u] XOR color[v] == voltage(u,v).
    # If we can assign colors satisfying all constraints, the lift is disconnected.
    # If not, it is connected.
    
    color = [-1] * n
    color[0] = 0
    stack = [0]
    
    while stack:
        u = stack.pop()
        for i, (v, w) in enumerate(adj_list[u]): # Need adj list
            vol = voltages[i]
            if color[w] == -1:
                color[w] = color[u] ^ vol
                stack.append(w)
            else:
                if (color[u] ^ color[w]) != vol:
                    # Inconsistent -> Connected
                    return True
                    
    # If we finish without inconsistency, it's consistent -> Disconnected
    return False
