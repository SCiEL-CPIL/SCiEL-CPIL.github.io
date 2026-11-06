"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        m = len(edges)
        # Build cost matrix: cost[i][j] = 0 if edge i can be immediately followed by edge j
        # (they share a vertex), BIG otherwise.
        # We want to maximize the number of such valid "chaining" links, which minimizes
        # the number of paths.
        BIG = 10**9
        cost = np.full((m, m), BIG, dtype=np.int64)
        
        # For each pair (i, j), check if edges[i] and edges[j] share a vertex and i != j
        for i in range(m):
            u_i, v_i = edges[i]
            for j in range(m):
                if i == j:
                    continue
                u_j, v_j = edges[j]
                # Check if they share a vertex
                if u_i == u_j or u_i == v_j or v_i == u_j or v_i == v_j:
                    cost[i][j] = 0
        
        # Find minimum cost assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Build next/prev pointers
        next_edge = [-1] * m
        prev_edge = [-1] * m
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] == 0:
                next_edge[r] = c
                prev_edge[c] = r
        
        # Find path starts
        paths = []
        for i in range(m):
            if prev_edge[i] == -1:
                # Start a new path at edge i
                path_edges = [i]
                cur = i
                while next_edge[cur] != -1:
                    cur = next_edge[cur]
                    path_edges.append(cur)
                
                # Convert edge indices to vertex list
                # Determine the vertex sequence by walking through the edges
                # For each consecutive pair, find the shared vertex
                if len(path_edges) == 1:
                    vertices = list(edges[path_edges[0]])
                else:
                    # Start with the first edge
                    u0, v0 = edges[path_edges[0]]
                    # The next edge shares a vertex with the first
                    u1, v1 = edges[path_edges[1]]
                    # Find the shared vertex between edge 0 and edge 1
                    if u0 == u1 or u0 == v1:
                        shared = u0
                    else:
                        shared = v0
                    # The path starts at the non-shared vertex of the first edge
                    if shared == u0:
                        vertices = [v0, u0]
                    else:
                        vertices = [u0, v0]
                    
                    # Continue with the rest of the edges
                    for k in range(1, len(path_edges)):
                        uk, vk = edges[path_edges[k]]
                        uk1, vk1 = edges[path_edges[k + 1]]
                        # Find shared vertex between edge k and edge k+1
                        if uk == uk1 or uk == vk1:
                            shared = uk
                        else:
                            shared = vk
                        # The next vertex in the path is the non-shared vertex of edge k+1
                        if shared == uk1:
                            vertices.append(vk1)
                        else:
                            vertices.append(uk1)
                
                paths.append(vertices)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    return results
# EVOLVE-BLOCK-END
