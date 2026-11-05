import numpy as np

def construct():
    # 1. Generate points of PG(2, 5)
    # Points are 1D subspaces of GF(5)^3
    # We can represent each point by a canonical vector (first non-zero entry is 1)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonical form: find first non-zero, divide by it
                vec = [x, y, z]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 4, 5) # Inverse in GF(5)
                        vec = [ (v * inv) % 5 for v in vec ]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    n = len(points) # Should be 31
    if n != 31:
        # Fallback if generation fails, though it shouldn't
        raise ValueError("Expected 31 points")

    # 2. Define Polarity
    # Bilinear form B(u, v) = u0*v0 + u1*v1 + u2*v2 mod 5
    # Two points are adjacent if B(u, v) == 0
    # Note: This polarity has absolute points where B(u, u) == 0
    # We construct the graph excluding loops (i.e., u != v)
    
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            u = np.array(points[i])
            v = np.array(points[j])
            prod = np.dot(u, v) % 5
            if prod == 0:
                A[i, j] = 1
                A[j, i] = 1
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append([i, j])
                
    # Current edge count should be 90
    if len(edges) != 90:
        # If it's not 90, we might need to adjust, but theory says 90
        # Let's proceed to add 1 edge
        pass

    # 3. Find best edge to add
    # We want to minimize the number of 4-cycles.
    # Adding edge (u, v) adds (A^3)[u, v] to the 4-cycle count?
    # Actually, the number of new 4-cycles is the number of length-3 paths between u and v.
    # Let's compute A^3
    A3 = np.linalg.matrix_power(A, 3)
    
    best_u = -1
    best_v = -1
    min_paths = float('inf')
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                # Count length-3 paths
                paths = A3[i, j]
                if paths < min_paths:
                    min_paths = paths
                    best_u = i
                    best_v = j
                    
    # Add the best edge
    if best_u != -1:
        edges.append([best_u, best_v])
        
    return edges
