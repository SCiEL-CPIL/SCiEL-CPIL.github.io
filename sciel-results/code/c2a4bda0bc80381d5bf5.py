"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    """
    Construct a graph with 31 vertices and 91 edges that minimizes the number of C4s.
    
    Strategy: 
    1. Try multiple polarity-based constructions to find a 90-edge C4-free base
    2. Add the edge that creates the fewest new C4s
    3. Perform local search (edge swaps) to minimize C4 count further
    """
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds safety margin (150s limit)
    
    n = 31
    q = 5
    
    def count_c4s(adj):
        """Count number of C4 subgraphs using vectorized operations."""
        # For each pair (u,v), C(common_neighbors(u,v), 2)
        # Each C4 is counted exactly twice (once for each pair of opposite vertices)
        c4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = np.dot(adj[u], adj[v])
                if common >= 2:
                    c4_count += common * (common - 1) // 2
        return c4_count // 2
    
    def local_search(A, deadline, max_iterations=5000):
        """Perform local search with edge swaps to minimize C4 count."""
        current_c4s = count_c4s(A)
        best_c4s = current_c4s
        best_A = A.copy()
        
        random.seed(42)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Get current edges and non-edges
            edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 1]
            non_edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 0]
            
            if not edge_list or not non_edge_list:
                break
            
            # Shuffle for randomness
            random.shuffle(edge_list)
            random.shuffle(non_edge_list)
            
            improved = False
            # Try a subset of swaps
            for e_out in edge_list[:30]:
                for e_in in non_edge_list[:30]:
                    if e_out == e_in:
                        continue
                    
                    # Try the swap
                    A[e_out[0], e_out[1]] = 0
                    A[e_out[1], e_out[0]] = 0
                    A[e_in[0], e_in[1]] = 1
                    A[e_in[1], e_in[0]] = 1
                    
                    new_c4s = count_c4s(A)
                    
                    if new_c4s < best_c4s:
                        best_c4s = new_c4s
                        best_A = A.copy()
                        improved = True
                    else:
                        # Revert
                        A[e_out[0], e_out[1]] = 1
                        A[e_out[1], e_out[0]] = 1
                        A[e_in[0], e_in[1]] = 0
                        A[e_in[1], e_in[0]] = 0
                    
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_A, best_c4s
    
    # Define points in PG(2, 5)
    points = []
    for t in range(5):
        points.append((1, t, 0))
    points.append((0, 1, 0))
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    
    assert len(points) == 31
    
    # Try multiple bilinear forms to find a 90-edge C4-free graph
    # Standard polarity: B(u,v) = u0*v0 + u1*v1 + u2*v2
    # We can also try other symmetric bilinear forms
    
    best_A = None
    best_c4s = float('inf')
    
    # Try the standard polarity
    bilinear_forms = [
        lambda u, v: (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q,  # standard
        lambda u, v: (u[0]*v[0] + 2*u[1]*v[1] + 3*u[2]*v[2]) % q,  # scaled
        lambda u, v: (u[0]*v[0] + u[1]*v[2] + u[2]*v[1]) % q,  # symmetric off-diagonal
    ]
    
    for bilinear in bilinear_forms:
        if time.time() > deadline - 30:
            break
        
        A = np.zeros((n, n), dtype=np.int8)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ui, vi, wi = points[i]
                uj, vj, wj = points[j]
                val = bilinear((ui, vi, wi), (uj, vj, wj))
                if val == 0:
                    A[i, j] = 1
        
        # Make symmetric and remove self-loops
        A = (A + A.T) // 2
        np.fill_diagonal(A, 0)
        
        edge_count = A.sum() // 2
        
        if edge_count == 90:
            # Find the non-edge that creates the fewest new C4s
            A3 = A @ A @ A
            
            min_new_c4s = float('inf')
            best_edge = None
            
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        c4s = A3[u, v]
                        if c4s < min_new_c4s:
                            min_new_c4s = c4s
                            best_edge = (u, v)
            
            if best_edge is not None:
                u, v = best_edge
                A[u, v] = 1
                A[v, u] = 1
                
                # Now do local search to minimize C4s
                A_optimized, c4s = local_search(A, deadline)
                
                if c4s < best_c4s:
                    best_c4s = c4s
                    best_A = A_optimized
        
        elif edge_count < 90:
            # Add edges until we have 90, then add one more
            # This is a fallback; try to add edges that minimize C4s
            A3 = A @ A @ A
            added = 0
            while edge_count < 90 and added < 10:
                min_new_c4s = float('inf')
                best_edge = None
                for u in range(n):
                    for v in range(u + 1, n):
                        if A[u, v] == 0:
                            c4s = A3[u, v]
                            if c4s < min_new_c4s:
                                min_new_c4s = c4s
                                best_edge = (u, v)
                if best_edge is None:
                    break
                u, v = best_edge
                A[u, v] = 1
                A[v, u] = 1
                edge_count += 1
                added += 1
                A3 = A @ A @ A
            
            if edge_count == 90:
                # Add one more edge
                A3 = A @ A @ A
                min_new_c4s = float('inf')
                best_edge = None
                for u in range(n):
                    for v in range(u + 1, n):
                        if A[u, v] == 0:
                            c4s = A3[u, v]
                            if c4s < min_new_c4s:
                                min_new_c4s = c4s
                                best_edge = (u, v)
                
                if best_edge is not None:
                    u, v = best_edge
                    A[u, v] = 1
                    A[v, u] = 1
                    
                    A_optimized, c4s = local_search(A, deadline)
                    
                    if c4s < best_c4s:
                        best_c4s = c4s
                        best_A = A_optimized
    
    # If no valid graph found, use fallback
    if best_A is None:
        # Fallback: random graph + local search
        A = np.zeros((n, n), dtype=np.int8)
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges[:91]:
            A[u, v] = 1
            A[v, u] = 1
        
        best_A, best_c4s = local_search(A, deadline)
    
    # Build edge list from best_A
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
    
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    
    for e in edges:
        assert 0 <= e[0] <= 30 and 0 <= e[1] <= 30
        assert e[0] != e[1]
    
    edge_set = set()
    for e in edges:
        key = tuple(sorted(e))
        assert key not in edge_set, f"Duplicate edge: {e}"
        edge_set.add(key)
    
    return edges

# EVOLVE-BLOCK-END
