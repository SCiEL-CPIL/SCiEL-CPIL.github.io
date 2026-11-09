import itertools
import time

def construct():
    """
    Construct a dense triangle-free graph on 16 vertices that is not 3-colorable.
    Strategy:
    1. Start with the Groetzsch graph (11 vertices, 20 edges, chi=4).
    2. Add 5 isolated vertices (12-16).
    3. Greedily add edges to maximize total edge count while maintaining triangle-free property.
    Since the base is not 3-colorable, the supergraph will also not be 3-colorable.
    """
    n = 16
    
    # Groetzsch graph edges (Mycielski of C5)
    # Vertices 0-4 form C5
    # Vertices 5-9 are copies
    # Vertex 10 is the root
    # Edges:
    # C5: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Copies: (5,6), (6,7), (7,8), (8,9), (9,5)
    # Connections: u_i to v_j if u_i u_j in C5
    # Root 10 connected to all v_i
    
    groetzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [5, 6], [6, 7], [7, 8], [8, 9], [9, 5],
        [0, 6], [0, 7],
        [1, 5], [1, 7],
        [2, 6], [2, 8],
        [3, 7], [3, 9],
        [4, 8], [4, 9],
        [5, 10], [6, 10], [7, 10], [8, 10], [9, 10]
    ]
    
    # Initialize adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in groetzsch_edges:
        adj[u][v] = True
        adj[v][u] = True
        
    edges = set(tuple(sorted((u, v))) for u, v in groetzsch_edges)
    
    # List of all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges:
                all_edges.append((u, v))
                
    # Greedily add edges
    # To maximize edges, we should prioritize edges that don't block many other edges?
    # Simple greedy: sort by some heuristic or just iterate.
    # A better heuristic: add edges that have the most "room" (fewest common neighbors).
    
    # We can use a simple local search:
    # 1. Add an edge if no triangle is formed.
    # 2. Repeat until no more edges can be added.
    
    changed = True
    while changed:
        changed = False
        # Shuffle to break symmetry? Or just iterate.
        for u, v in all_edges:
            if (u, v) in edges:
                continue
            # Check if adding (u, v) creates a triangle
            # Triangle exists if there is w such that adj[u][w] and adj[v][w]
            creates_triangle = False
            for w in range(n):
                if w != u and w != v:
                    if adj[u][w] and adj[v][w]:
                        creates_triangle = True
                        break
            if not creates_triangle:
                # Add edge
                adj[u][v] = True
                adj[v][u] = True
                edges.add((u, v))
                changed = True
                
    # Convert to list of lists
    result = [list(e) for e in edges]
    return result
