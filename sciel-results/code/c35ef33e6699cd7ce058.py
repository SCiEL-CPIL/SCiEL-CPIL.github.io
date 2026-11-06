import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # 1. Generate points of PG(2,5)
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: first non-zero component is 1
                if x != 0:
                    inv_x = pow(x, 4, 5)
                    pts = (1, (y * inv_x) % 5, (z * inv_x) % 5)
                elif y != 0:
                    inv_y = pow(y, 4, 5)
                    pts = (0, 1, (z * inv_y) % 5)
                else:
                    pts = (0, 0, 1)
                if pts not in points:
                    points.append(pts)
    
    n = len(points)
    if n != 31:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    # 2. Build adjacency matrix based on polarity B(u,v) = u1*v1 + u2*v2 + u3*v3 = 0
    A = np.zeros((n, n), dtype=np.int64)
    
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            b_val = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if b_val == 0:
                A[i, j] = 1
                A[j, i] = 1
    
    edge_count = int(A.sum()) // 2
    if edge_count != 90:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    # 3. Compute A^3 to find best edge to add
    # For a C4-free graph, C4s created by adding non-edge (i,j) = A^3[i,j]
    A2 = A @ A
    A3 = A @ A2
    
    # Find non-edge with minimum A3[i,j]
    min_val = float('inf')
    best_i, best_j = -1, -1
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if A[i, j] == 1:
                continue
            val = int(A3[i, j])
            if val < min_val:
                min_val = val
                best_i, best_j = i, j
    
    if best_i == -1:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    # 4. Build edge list and adjacency structure
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append([i, j])
    
    edges.append([best_i, best_j])
    A[best_i, best_j] = 1
    A[best_j, best_i] = 1
    
    # 5. Local search: try to improve by swapping edges
    # Count C4s efficiently: C4 count = 0.5 * sum_{u,v} (common_neighbors(u,v))^2
    # Actually, standard count: sum over all pairs {u,v} of C(common(u,v), 2)
    # Let's use the formula: C4_count = 1/4 * sum_{u,v} ( (A^2)[u,v] choose 2 )? No.
    # C4 count = sum_{u < v} C(c(u,v), 2) where c(u,v) is common neighbors.
    
    def count_c4(A_mat):
        A2_mat = A_mat @ A_mat
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = A2_mat[i, j]
                if c >= 2:
                    total += c * (c - 1) // 2
        return total
    
    current_c4 = count_c4(A)
    best_edges = [e[:] for e in edges]
    best_c4 = current_c4
    
    # Precompute neighbor lists for faster common neighbor checks if needed,
    # but A^2 is fast enough with NumPy for n=31.
    
    # Local search loop
    # Try removing one edge and adding another.
    # We have 91 edges.
    # There are 31*30/2 = 465 possible edges.
    # Non-edges = 465 - 91 = 374.
    
    # To speed up, we can iterate over current edges to remove, and candidate non-edges to add.
    # But 91 * 374 ~ 34k swaps. Each swap involves updating A and counting C4s.
    # Counting C4s takes O(n^3) ~ 30k ops.
    # 34k * 30k = 1e9 ops. Too slow for Python loops if not vectorized.
    
    # Optimization:
    # When we remove edge (u,v) and add edge (x,y), how does C4 count change?
    # This is complex. Let's just do a limited number of random or structured swaps if time permits,
    # or stick to the greedy initialization which is likely optimal or near-optimal.
    
    # Given the time limit (100s) and the fact that the greedy step is very fast,
    # we have plenty of time for local search.
    # However, full C4 recount is O(n^3).
    # Let's try a few iterations of 1-move swaps (remove one, add one).
    
    # We will sample a subset of swaps to stay within time limits if needed,
    # but n=31 is small.
    
    # Let's try all 1-edge swaps (remove e, add f) where e is in edges, f is not in edges.
    # We need to be careful with the "add" part.
    
    # Strategy:
    # 1. List current edges.
    # 2. List non-edges.
    # 3. For each edge e to remove:
    #    - Remove e from A.
    #    - For each non-edge f to add:
    #      - Add f to A.
    #      - Count C4s.
    #      - If better, record.
    #      - Remove f from A.
    #    - Add e back to A.
    
    # This is O(E * (E_non * n^3)).
    # E = 91, E_non = 374.
    # 91 * 374 * 30000 ~ 1e9. This might be too slow in pure Python even with NumPy inside.
    # count_c4 uses A @ A which is O(n^3) but optimized C code.
    # 34000 matrix multiplications of 31x31 matrices.
    # 31^3 = ~30k flops.
    # 34000 * 30k = 1e9 flops.
    # NumPy matmul for 31x31 is very fast, maybe 1-2 microseconds?
    # 34000 * 2us = 68ms. This is very fast!
    # The overhead is the Python loop.
    # 34000 iterations in Python is fast (<0.1s).
    # So we can do a full local search.
    
    edge_set = set(map(tuple, edges))
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                non_edges.append((i, j))
    
    # Sort non_edges by some heuristic? Maybe just iterate.
    
    improved = True
    while improved:
        improved = False
        best_swap = None
        best_new_c4 = best_c4
        
        # Iterate over edges to remove
        for rem_idx in range(len(edges)):
            if time.time() > deadline - 10:
                break
            
            u, v = edges[rem_idx]
            
            # Remove edge
            A[u, v] = 0
            A[v, u] = 0
            
            # Iterate over non-edges to add
            # Optimization: skip non-edges that are the same as removed edge (obviously)
            for add_i, add_j in non_edges:
                if add_i == u and add_j == v:
                    continue
                if add_i == v and add_j == u:
                    continue
                
                # Add edge
                A[add_i, add_j] = 1
                A[add_j, add_i] = 1
                
                new_c4 = count_c4(A)
                
                # Remove edge
                A[add_i, add_j] = 0
                A[add_j, add_i] = 0
                
                if new_c4 < best_new_c4:
                    best_new_c4 = new_c4
                    best_swap = (rem_idx, (add_i, add_j))
            
            # Restore removed edge
            A[u, v] = 1
            A[v, u] = 1
            
            if time.time() > deadline - 10:
                break
        
        if best_swap is not None:
            rem_idx, (add_i, add_j) = best_swap
            rem_u, rem_v = edges[rem_idx]
            
            # Update edges list
            edges.pop(rem_idx)
            edges.append([add_i, add_j])
            
            # Update A
            A[rem_u, rem_v] = 0
            A[rem_v, rem_u] = 0
            A[add_i, add_j] = 1
            A[add_j, add_i] = 1
            
            # Update non_edges list (remove added, add removed)
            # This is O(E) per update, which is fine.
            non_edges.remove((add_i, add_j))
            non_edges.append((rem_u, rem_v))
            
            best_c4 = best_new_c4
            best_edges = [e[:] for e in edges]
            improved = True
    
    return best_edges
