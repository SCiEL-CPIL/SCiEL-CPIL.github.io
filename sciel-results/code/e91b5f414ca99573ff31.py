import numpy as np
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # For the specific case of k=4, g=10, use d=4 to minimize vertices.
    # General rule: d = max(1, (g - 2) // 2 - 1)
    d = max(1, (g - 2) // 2 - 1)
    
    q = k
    
    # LUW Construction Parameters
    # Vertices are (side, x_1, ..., x_{d+1})
    # We fix x_1 = 0 to reduce symmetry? No, standard LUW uses all.
    # Standard LUW has 2 * q^d vertices if we fix one coordinate?
    # Let's use the BFS approach from the seed but with correct d.
    
    # Precompute rule table for LUW
    # The rules depend on d and q.
    # For q=4, we use GF(4).
    
    if q == 4:
        # GF(4) table
        gf4_add = np.array([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]], dtype=np.int64)
        gf4_mul = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        gf4_add = None
        gf4_mul = None

    # Start vertex
    start_vertex = (0, tuple([0] * (d + 1)))
    visited = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    vertex_idx = 1
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
            
        side, x_tuple = vertex
        x = np.array(x_tuple, dtype=np.int64)
        
        # Generate q neighbors
        # y[0] is free? No, in LUW, one coordinate is determined by the others.
        # Standard LUW: 
        # Neighbors of (0, x) are (1, y) where y[0] = 0 and for j in 1..d:
        # y[j] = x[j] + x[j-1] * y[j-1] ? 
        # This specific recurrence ensures girth.
        
        # Let's use the specific recurrence for LUW cages.
        # y[0] = 0
        # For j = 1 to d:
        #   y[j] = x[j] + x[j-1] * y[j-1] (for side 0 -> 1)
        #   y[j] = x[j] - x[j-1] * y[j-1] (for side 1 -> 0)
        
        Y = np.zeros((q, d + 1), dtype=np.int64)
        Y[:, 0] = 0 # Fix first coordinate
        
        # We need q distinct neighbors. 
        # In LUW, the "free" variable is usually y[d] or similar?
        # Actually, the map x -> y is bijective if the recurrence is correct.
        # But we need k=q neighbors.
        # The standard LUW construction defines edges via:
        # (x, 0) ~ (y, 1) iff y = A x + b?
        
        # Let's stick to the BFS generation from the seed, which seemed to work for some cases,
        # but ensure d is minimal.
        
        # Seed logic:
        # Y[:, 1] = 0..q-1
        # for c in 2..d: ...
        
        # I will replicate the seed's neighbor generation but with d=4.
        
        Y[:, 1] = np.arange(q)
        
        for c in range(2, d + 1):
            # Rule from seed
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            # This rule table in the seed seems specific to a certain variant.
            # For a robust LUW, we use:
            # y[c] = x[c] + x[c-1] * y[c-1]
            
            if side == 0:
                if q == 4:
                    prod = gf4_mul[x[c-1], Y[:, c-1]]
                    Y[:, c] = gf4_add[x[c], prod]
                else:
                    Y[:, c] = (x[c] + x[c-1] * Y[:, c-1]) % q
            else:
                if q == 4:
                    prod = gf4_mul[x[c-1], Y[:, c-1]]
                    Y[:, c] = gf4_add[x[c], prod] # Subtraction in GF(4) is addition
                else:
                    Y[:, c] = (x[c] - x[c-1] * Y[:, c-1]) % q
                    
        for j in range(q):
            neighbor = (1 - side, tuple(Y[j]))
            
            if neighbor not in visited:
                visited[neighbor] = vertex_idx
                order.append(neighbor)
                vertex_idx += 1
            
            if side == 0:
                edges.append((visited[vertex], visited[neighbor]))
                
    return len(order), edges
