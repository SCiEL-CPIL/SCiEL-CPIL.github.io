import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    q = 5
    n = 31
    
    # 1. Generate points of PG(2,5)
    # 25 affine points (x, y, 1)
    # 5 points at infinity (1, t, 0) for t=0..4
    # 1 point at infinity (0, 1, 0)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for t in range(q):
        points.append((1, t, 0))
    points.append((0, 1, 0))
    
    if len(points) != 31:
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
    
    # 3. Compute A^3 to find best non-edge to add
    # For a C4-free graph, adding non-edge (i,j) creates A3[i,j] four-cycles
    A2 = A @ A
    A3 = A @ A2
    
    # Find non-edge with minimum A3[i,j]
    min_val = float('inf')
    best_i, best_j = -1, -1
    
    for i in range(n):
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
    
    # 4. Add the best edge
    A[best_i, best_j] = 1
    A[best_j, best_i] = 1
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append([i, j])
    
    # 5. Local search: try edge swaps to reduce C4 count
    # Count C4s using matrix formula: C4 = (1/8) * sum_{i,j} (A^2[i,j]) * (A^2[i,j]-1) for i != j
    # More precisely: number of C4s = (1/8) * trace(A^4) - (1/2) * sum_{i,j} A[i,j] * (A^2[i,j] - A[i,j])
    # Actually: C4 count = (1/8) * (trace(A^4) - 2*sum(A^2) + 4*sum(A))
    # But simpler: for each pair of vertices, count pairs of common neighbors
    # C4 = sum over unordered pairs {u,v} of C(common_neighbors(u,v), 2)
    
    def count_c4_matrix(Adj):
        """Count C4s using adjacency matrix."""
        A2 = Adj @ Adj
        # For each pair (i,j), the number of common neighbors is A2[i,j]
        # Number of C4s = sum_{i<j} C(A2[i,j], 2) = sum_{i<j} A2[i,j]*(A2[i,j]-1)/2
        # But this counts each C4 exactly once if we're careful
        # Actually each C4 has 4 pairs of opposite vertices, each contributing 1 common neighbor pair
        # Wait, let me think again.
        # A C4 is u-v-w-x-u. The pairs of opposite vertices are (u,w) and (v,x).
        # For pair (u,w), v and x are both common neighbors, so C2=1.
        # For pair (v,x), u and w are both common neighbors, so C2=1.
        # So each C4 is counted exactly twice in sum_{i<j} C(A2[i,j],2).
        # Therefore C4 = (1/2) * sum_{i<j} A2[i,j]*(A2[i,j]-1)
        
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = int(A2[i, j])
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2
    
    current_c4 = count_c4_matrix(A)
    best_c4 = current_c4
    best_A = A.copy()
    
    # Try local search: remove one edge, add another non-edge
    # Precompute A2 for current graph
    A2_current = A @ A
    
    for iteration in range(50):
        if time.time() > deadline - 15:
            break
        
        improved = False
        
        # Try removing each edge and finding best replacement
        for rem_idx in range(len(edges)):
            if time.time() > deadline - 15:
                break
            
            u, v = edges[rem_idx]
            
            # Remove edge
            A[u, v] = 0
            A[v, u] = 0
            
            # Recompute A2 and A3 after removal
            A2_new = A @ A
            A3_new = A @ A2_new
            
            # Find best non-edge to add (not the one we just removed)
            min_add_val = float('inf')
            best_add_i, best_add_j = -1, -1
            
            for i in range(n):
                for j in range(i + 1, n):
                    if A[i, j] == 1:
                        continue
                    if i == u and j == v:
                        continue
                    val = int(A3_new[i, j])
                    if val < min_add_val:
                        min_add_val = val
                        best_add_i, best_add_j = i, j
            
            if best_add_i != -1 and min_add_val < 100:  # Only consider if it's good enough
                # Add the edge
                A[best_add_i, best_add_j] = 1
                A[best_add_j, best_add_i] = 1
                
                new_c4 = count_c4_matrix(A)
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_A = A.copy()
                    improved = True
                else:
                    # Undo
                    A[best_add_i, best_add_j] = 0
                    A[best_add_j, best_add_i] = 0
            
            # Restore removed edge
            A[u, v] = 1
            A[v, u] = 1
        
        if not improved:
            break
    
    # Build final edge list from best_A
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j] == 1:
                final_edges.append([i, j])
    
    if len(final_edges) != 91:
        # Fallback to the initial construction
        return edges
    
    return final_edges
