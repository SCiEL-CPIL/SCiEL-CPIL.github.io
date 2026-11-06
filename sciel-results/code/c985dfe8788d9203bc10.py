import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    # adj[i][j] is True if edge exists
    adj = [[False] * n for _ in range(n)]
    
    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True
        
    # 1. Grötzsch Graph Initialization (Vertices 0-10)
    # 0-4: Cycle
    for i in range(5):
        add_edge(i, (i + 1) % 5)
        
    # 5-9: Shadows
    # Shadow of i is 5+i
    # Shadow of i connects to neighbors of i in cycle
    for i in range(5):
        s = 5 + i
        add_edge(s, (i + 1) % 5)
        add_edge(s, (i - 1) % 5)
        
    # 10: Top connected to all shadows
    for i in range(5):
        add_edge(10, 5 + i)
        
    # 2. Add Vertex 11 connected to 10, 0, 2
    # {10, 0, 2} is an independent set in Grötzsch
    add_edge(11, 10)
    add_edge(11, 0)
    add_edge(11, 2)
    
    # Current Vertices: 0-11 (12 vertices)
    # Current Edges: 20 (Grötzsch) + 3 = 23
    # Max Degree: Vertex 10 has degree 6 (5 from Grötzsch + 1 from 11)
    
    # 3. Expand Vertex 10 into 5 vertices: 10, 12, 13, 14, 15
    # This adds 4 vertices (12, 13, 14, 15)
    # Total vertices: 16
    # Vertices 10, 12, 13, 14, 15 form an independent set
    # They are all connected to the neighbors of 10 in the 12-vertex graph
    # Neighbors of 10: 5, 6, 7, 8, 9, 11
    
    neighbors_of_10 = [5, 6, 7, 8, 9, 11]
    expansion_vertices = [10, 12, 13, 14, 15]
    
    # Remove original edges from 10? 
    # Actually, we just add edges from all expansion vertices to neighbors.
    # The original edges from 10 are a subset of these.
    # So we can just iterate and add.
    
    for v in expansion_vertices:
        for u in neighbors_of_10:
            add_edge(v, u)
            
    # Note: add_edge is idempotent, so re-adding 10's edges is fine.
    # But we must ensure we don't add edges between expansion vertices.
    # They are independent, so no edges between them.
    
    # Construct edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges
