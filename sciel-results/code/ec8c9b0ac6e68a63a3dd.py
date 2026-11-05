import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, leaving 20s margin
    
    n = 31
    # Generate points in PG(2, 5)
    # Representation: (x, y, z) in F_5^3
    points = []
    
    # Points with z=0
    # (1,0,0), (1,1,0), (1,2,0), (1,3,0), (1,4,0), (0,1,0)
    points.append((1, 0, 0))
    points.append((1, 1, 0))
    points.append((1, 2, 0))
    points.append((1, 3, 0))
    points.append((1, 4, 0))
    points.append((0, 1, 0))
    
    # Points with z=1
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
            
    # Build adjacency matrix A
    # Polarity defined by B(u,v) = u0*v0 + u1*v1 + u2*v2 == 0 (mod 5)
    A = np.zeros((n, n), dtype=np.int32)
    edges = []
    
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            # Dot product mod 5
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if dot == 0:
                A[i, j] = 1
                A[j, i] = 1
                edges.append([i, j])
                
    # The theory predicts 90 edges. 
    # If it's not 90, we might have an issue with the form, 
    # but for q=5, I3 is a valid non-degenerate form.
    
    if time.time() > deadline:
        # Should not happen, but return what we have if we need to pad?
        # If we have <91, we can't pad correctly. 
        # But this step is <1ms.
        pass

    # Compute A^3 to count paths of length 3
    A3 = np.linalg.matrix_power(A, 3)
    
    # Find the non-edge that minimizes the number of new 4-cycles
    # which is the number of length-3 paths between the endpoints
    min_paths = float('inf')
    best_edge = None
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if A[i, j] == 0:
                paths = A3[i, j]
                if paths < min_paths:
                    min_paths = paths
                    best_edge = [i, j]
                    
    if best_edge is None:
        # Fallback if no non-edge found (complete graph? impossible with 90 edges)
        # Just pick any pair
        best_edge = [0, 1]
        
    edges.append(best_edge)
    
    return edges
