"""LUW construction for 4-regular cage graphs (girth 10)."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s/150s kill
    
    q = k  # For k=4, q=4 (GF(4))
    # Minimum d for girth >= g: d = g - 5 (from LUW theory)
    d = max(3, g - 5)
    
    # Precompute rule table for LUW construction
    # The rule determines how coordinates propagate in the Cayley graph
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # GF(4) multiplication table
    gf4_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    gf4_add = gf4_table  # XOR for addition in GF(4)
    
    # Build the graph using BFS to enumerate all vertices
    # Vertices are (side, x_tuple) where x_tuple is a tuple of length d+1
    # side=0: "point" vertices, side=1: "line" vertices
    
    start_vertex = (0, (0,) * (d + 1))
    visited = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    vertex_idx = 1
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        
        side, x_tuple = vertex
        x = np.array(x_tuple, dtype=np.int64)
        
        # Generate all q neighbors
        # Y has shape (q, d+1), Y[:, 1] = 0..q-1
        Y = np.zeros((q, d + 1), dtype=np.int64)
        Y[:, 1] = np.arange(q)
        
        for c in range(2, d + 1):
            a, b = rule[c]
            if side == 0:
                # Point -> Line: y[c] = x[c] + y[a] * x[b] in GF(4)
                prod = gf4_table[Y[:, a], x[b]]
                Y[:, c] = prod ^ x[c]
            else:
                # Line -> Point: y[c] = x[c] - x[a] * y[b] in GF(4)
                # In GF(4), subtraction is same as addition (characteristic 2)
                prod = gf4_table[x[a], Y[:, b]]
                Y[:, c] = prod ^ x[c]
        
        for j in range(q):
            neighbor = (1 - side, tuple(Y[j]))
            
            if neighbor not in visited:
                visited[neighbor] = vertex_idx
                order.append(neighbor)
                vertex_idx += 1
            
            # Add edge (avoid duplicates by only adding when side=0)
            if side == 0:
                edges.append((visited[vertex], visited[neighbor]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
