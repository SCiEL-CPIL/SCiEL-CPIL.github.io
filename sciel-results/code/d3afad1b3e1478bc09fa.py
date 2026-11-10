import numpy as np

def construct():
    # Generate points of P^2(F_5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonicalize
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    cx = 1
                    cy = (y * inv_x) % 5
                    cz = (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    cx = 0
                    cy = 1
                    cz = (z * inv_y) % 5
                else:
                    cx = 0
                    cy = 0
                    cz = 1
                
                p = (cx, cy, cz)
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    n = len(points)
    # n should be 31
    
    # Build adjacency matrix
    # A[i][j] = 1 if edge exists
    A = np.zeros((n, n), dtype=int)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            # Bilinear form B(p1, p2) = p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]
            b = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if b == 0:
                A[i][j] = 1
                A[j][i] = 1
                edges.append([i, j])
                
    # Check edge count
    # It should be 90
    if len(edges) != 90:
        # Fallback to random if construction fails (should not happen)
        # But let's assume it works
        pass
        
    # Find the best edge to add
    # For each non-edge (u, v), count paths of length 3
    # Path u-a-b-v
    # Count = sum_{a in N(u)} |N(a) \cap N(v)| ? No.
    # Number of a such that (u,a) in E and (a,b) in E and (b,v) in E.
    # This is equivalent to (A^3)[u][v].
    # A^3 counts walks of length 3.
    # Since the graph is simple, walks of length 3 are just paths if we don't care about backtracking?
    # A walk u-a-b-v can have a=v? No, simple graph, no self loops.
    # Can a=u? No.
    # Can b=u? u-a-u-v? No, (u,u) not in E.
    # Can a=b? u-a-a-v? No, (a,a) not in E.
    # So all walks of length 3 are simple paths u-a-b-v with distinct vertices?
    # Wait, u-a-b-v. If a=v, u-v-b-v? (v,v) no.
    # If b=u, u-a-u-v? (u,u) no.
    # If a=u, no.
    # If u=v, no.
    # So yes, (A^3)[u][v] is the number of length-3 paths between u and v.
    
    A3 = np.linalg.matrix_power(A, 3)
    
    best_edge = None
    min_paths = float('inf')
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] == 0:
                paths = A3[i][j]
                if paths < min_paths:
                    min_paths = paths
                    best_edge = [i, j]
                    
    edges.append(best_edge)
    
    return edges
