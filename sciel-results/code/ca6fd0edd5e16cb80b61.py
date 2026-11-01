import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # LUV graph CD(D, q)
    # q = k
    # D = g - 5
    if k == 4 and g == 10:
        q = 4
        D = 5
    else:
        # Fallback to seed logic if not the specific case
        # But we assume k=4, g=10
        q = k
        D = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
    # GF(q) arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Vertices: 0..q^(D-1)-1 are points, q^(D-1)..2*q^(D-1)-1 are lines
    # Point index: tuple of D-1 values
    # Line index: tuple of D-1 values
    
    n_vertices = 2 * (q ** (D - 1))
    point_count = q ** (D - 1)
    
    # Map point tuple to index
    point_indices = {}
    for i in range(point_count):
        # Convert i to base q, length D-1
        coords = []
        val = i
        for _ in range(D - 1):
            coords.append(val % q)
            val //= q
        point_indices[tuple(coords)] = i
        
    line_indices = {}
    for i in range(point_count):
        coords = []
        val = i
        for _ in range(D - 1):
            coords.append(val % q)
            val //= q
        line_indices[tuple(coords)] = point_count + i
        
    edges = []
    
    # Iterate over all points
    for p_idx in range(point_count):
        x = []
        val = p_idx
        for _ in range(D - 1):
            x.append(val % q)
            val //= q
        x = tuple(x)
        
        # Iterate over all y_0
        for y0 in range(q):
            # Calculate y
            y = [0] * D  # y[0]..y[D-1]
            y[0] = y0
            for i in range(1, D):
                # y_i = x_i + x_{i+1} y_{i-1}
                # x is 1-indexed in formula, 0-indexed in list
                # x_i is x[i-1]
                # x_{i+1} is x[i] if i < D-1, else 0
                xi = x[i - 1]
                x_next = x[i] if i < D - 1 else 0
                y[i] = add(xi, mul(x_next, y[i - 1]))
            
            y_tuple = tuple(y[1:D])  # y_1..y_{D-1}
            l_idx = line_indices[y_tuple]
            
            # Add edge (p_idx, l_idx)
            edges.append((p_idx, l_idx))
            
            if time.time() > deadline:
                # Return what we have? No, must be valid.
                # If we break, the graph is not regular.
                # So we must finish or return a known valid graph.
                # Given the small size (512 vertices, 2048 edges), it should be fast.
                pass

    return n_vertices, edges
