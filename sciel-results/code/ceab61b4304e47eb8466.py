import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Generate PG(2,5) points
    points = []
    # Type 1: (1, y, z)
    for y in range(5):
        for z in range(5):
            points.append([1, y, z])
    # Type 2: (0, 1, z)
    for z in range(5):
        points.append([0, 1, z])
    # Type 3: (0, 0, 1)
    points.append([0, 0, 1])
    
    P = np.array(points, dtype=np.int64)
    n = 31
    
    # Polarity: B(u, v) = u1*v1 + (u2+u3)*(v2+v3) mod 5
    u_sum = (P[:, 1] + P[:, 2]) % 5
    u_1 = P[:, 0]
    
    M = np.zeros((n, n), dtype=np.int8)
    
    # Compute adjacency
    for i in range(n):
        for j in range(i+1, n):
            val = (u_1[i] * u_1[j] + u_sum[i] * u_sum[j]) % 5
            if val == 0:
                M[i, j] = 1
                M[j, i] = 1
                
    # Verify edge count is 90
    # edge_count = np.sum(M) // 2
    # assert edge_count == 90, f"Expected 90 edges, got {edge_count}"
    
    # Count C4s in current graph (should be 0)
    M2 = M @ M
    sum_sq = np.sum(M2**2) - np.sum(np.diag(M2)**2)
    c4_count = sum_sq // 8
    
    # Find best edge to add to minimize new C4s
    # New C4s = common neighbors of (i, j)
    best_val = 999
    best_i = -1
    best_j = -1
    
    for i in range(n):
        for j in range(i+1, n):
            if M[i, j] == 0:
                val = M2[i, j]
                if val < best_val:
                    best_val = val
                    best_i = i
                    best_j = j
                    if best_val == 0:
                        break
        if best_val == 0:
            break
            
    # Add the best edge
    if best_i != -1:
        M[best_i, best_j] = 1
        M[best_j, best_i] = 1
    else:
        # Fallback if something went wrong (should not happen)
        # Add first non-edge
        for i in range(n):
            for j in range(i+1, n):
                if M[i, j] == 0:
                    M[i, j] = 1
                    M[j, i] = 1
                    break
            if M[i, j] == 1 and i != j and M[i, j] == 1 and M[j, i] == 1 and not (i==0 and j==1 and M[0,1]==1): # crude check
                break

    # Construct edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if M[i, j] == 1:
                edges.append([i, j])
                
    # Ensure exactly 91 edges
    # If for some reason we have more or less, we might need to adjust, but logic ensures 90+1=91
    
    return edges
