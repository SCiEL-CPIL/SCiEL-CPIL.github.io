import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin before 150s kill
    
    # Field GF(5)
    # Points of PG(2, 5)
    # (x, y, z)
    points = []
    # Affine part: z=1
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    # Line at infinity: z=0
    # (1, m, 0) for m in 0..4
    for m in range(5):
        points.append((1, m, 0))
    # (0, 1, 0)
    points.append((0, 1, 0))
    
    n = len(points)
    # n should be 31
    
    # Adjacency matrix
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check orthogonality
            # x_i x_j + y_i y_j + z_i z_j == 0 mod 5
            val = (points[i][0] * points[j][0] + 
                   points[i][1] * points[j][1] + 
                   points[i][2] * points[j][2]) % 5
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append([i, j])
                
    # Should have 90 edges
    # The polarity graph of PG(2,5) is C4-free, so any non-edge has <=1 common neighbor
    # Adding any non-edge creates 0 C4s. But let's verify and pick the best one.
    
    # For each non-edge, compute number of common neighbors
    # In a C4-free graph, this should be 0 or 1 for all non-edges
    # If any non-edge has 2+ common neighbors, the base graph is not C4-free
    
    min_common = float('inf')
    best_pair = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                common = int(np.sum(A[i] * A[j]))
                if common < min_common:
                    min_common = common
                    best_pair = (i, j)
                    
    edges.append([best_pair[0], best_pair[1]])
    
    # Verify C4 count should be 0 if base graph is C4-free
    # Quick verification: count C4s
    c4_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            common = int(np.sum(A[i] * A[j]))
            if common >= 2:
                c4_count += common * (common - 1) // 2
    c4_count //= 2
    
    if c4_count == 0:
        # Perfect, return immediately
        return edges
    
    # If base graph is not C4-free, do local search
    # Fast C4 count using matrix multiplication
    def count_c4s_fast(adj):
        A2 = adj @ adj
        # (A2)[i,j] = number of common neighbors of i and j
        # C4 count = (1/2) * sum_{i<j} C(A2[i,j], 2)
        c4 = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = A2[i, j]
                if c >= 2:
                    c4 += c * (c - 1) // 2
        return c4 // 2
    
    current_c4s = count_c4s_fast(A)
    
    # Local search: try edge swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_swap = None
        best_c4s = current_c4s
        
        for e_idx in range(len(edges)):
            if time.time() > deadline:
                break
            u, v = edges[e_idx]
            
            # Remove edge
            A[u, v] = 0
            A[v, u] = 0
            
            # Try adding each non-edge
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if A[i, j] == 0 and not (i == u and j == v) and not (i == v and j == u):
                        # Quick estimate: adding edge (i,j) creates C(common(i,j),2) C4s
                        # where common(i,j) is computed in the graph without edge (u,v)
                        common_ij = int(np.sum(A[i] * A[j]))
                        # But we need to account for the fact that removing (u,v) might change things
                        # For a quick heuristic, just use the common count
                        new_c4s = current_c4s - 0  # rough
                        # More accurately: C4s involving edge (u,v) are removed, C4s involving (i,j) are added
                        # This is complex; use full count only for promising candidates
                        if common_ij <= 1:
                            # Likely good swap
                            A[i, j] = 1
                            A[j, i] = 1
                            c4s_with_add = count_c4s_fast(A)
                            A[i, j] = 0
                            A[j, i] = 0
                            if c4s_with_add < best_c4s:
                                best_c4s = c4s_with_add
                                best_swap = (e_idx, i, j)
            
            # Restore edge
            A[u, v] = 1
            A[v, u] = 1
        
        if best_swap is not None:
            e_idx, new_i, new_j = best_swap
            old_u, old_v = edges[e_idx]
            edges[e_idx] = [new_i, new_j]
            A[old_u, old_v] = 0
            A[old_v, old_u] = 0
            A[new_i, new_j] = 1
            A[new_j, new_i] = 1
            current_c4s = best_c4s
            improved = True
    
    return edges
