import numpy as np

def construct():
    n = 31
    # Define points in PG(2, 5)
    # Indices 0-5: z=0
    # Indices 6-30: z=1
    points = []
    
    # z=0 points: (x, y, 0)
    # (1,0), (1,1), (1,2), (1,3), (1,4), (0,1)
    pts_z0 = [
        (1, 0, 0),
        (1, 1, 0),
        (1, 2, 0),
        (1, 3, 0),
        (1, 4, 0),
        (0, 1, 0)
    ]
    
    # z=1 points: (x, y, 1)
    for i in range(5):
        for j in range(5):
            pts_z0.append((i, j, 1))
            
    points = pts_z0
    
    # Build adjacency matrix
    # Bilinear form: u0*v0 + u1*v1 + u2*v2 == 0 mod 5
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            val = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
                
    # Count edges to verify
    edge_count = sum(A.sum(axis=0)) // 2
    # Should be 90
    
    # Compute A^3
    # A^3[i, j] gives number of walks of length 3 from i to j
    A3 = np.linalg.matrix_power(A, 3)
    
    # Find the non-edge (i, j) with minimum A3[i, j]
    min_val = float('inf')
    best_pair = (-1, -1)
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                # i and j are not connected
                # Check if adding edge creates few C4s
                # A3[i, j] is the count of paths of length 3
                # Since A[i, j] == 0, A3[i, j] counts valid C4 candidates
                if A3[i, j] < min_val:
                    min_val = A3[i, j]
                    best_pair = (i, j)
                    
    # Add the best edge
    u, v = best_pair
    A[u, v] = 1
    A[v, u] = 1
    
    # Construct list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append([i, j])
                
    return edges
