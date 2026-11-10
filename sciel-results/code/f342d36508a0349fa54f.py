import numpy as np

def construct():
    # Generate all 31 points of PG(2, 5)
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    pts = (1, (y * inv_x) % 5, (z * inv_x) % 5)
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    pts = (0, 1, (z * inv_y) % 5)
                else:
                    pts = (0, 0, 1)
                if pts not in points:
                    points.append(pts)
    
    point_to_idx = {p: i for i, p in enumerate(points)}
    n = 31
    
    # Polarity matrix M = diag(1, 1, 4)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            val = (u[0]*v[0] + u[1]*v[1] + 4*u[2]*v[2]) % 5
            if val == 0:
                edges.append([i, j])
                
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.int32)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
        
    # Compute A^3 to find best edge to add
    A3 = np.linalg.matrix_power(A, 3)
    
    # Find the non-edge with minimum (A^3)[u][v]
    min_val = float('inf')
    best_edge = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                # Number of walks of length 3 from i to j
                val = A3[i, j]
                if val < min_val:
                    min_val = val
                    best_edge = [i, j]
                    
    # Add the best edge
    if best_edge:
        edges.append(best_edge)
        
    # Ensure exactly 91 edges
    # If for some reason we didn't add one (shouldn't happen), add any
    if len(edges) < 91:
        edge_set = set((min(u,v), max(u,v)) for u,v in edges)
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edge_set:
                    edges.append([i, j])
                    if len(edges) == 91:
                        break
            if len(edges) == 91:
                break
                
    return edges
