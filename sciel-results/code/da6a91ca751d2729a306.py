import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n = 31
    m = 91
    
    # Generate points of PG(2, 5)
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                if a != 0:
                    inv_a = pow(a, -1, 5)
                    p = (1, (b * inv_a) % 5, (c * inv_a) % 5)
                elif b != 0:
                    inv_b = pow(b, -1, 5)
                    p = (0, 1, (c * inv_b) % 5)
                else:
                    p = (0, 0, 1)
                
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Construct polarity graph
    adj = np.zeros((n, n), dtype=np.uint8)
    edges_set = set()
    
    for i in range(n):
        for j in range(i+1, n):
            p_i = points[i]
            p_j = points[j]
            dot = (p_i[0]*p_j[0] + p_i[1]*p_j[1] + p_i[2]*p_j[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges_set.add((i, j))
    
    # Count C4s efficiently using matrix multiplication
    def count_c4(adj):
        # C4 count = 1/2 * sum over all pairs (u,v) of C(common_neighbors(u,v), 2)
        # = 1/2 * (sum_{u,v} cn(u,v)^2 - sum_{u,v} cn(u,v))
        # where cn(u,v) = number of common neighbors
        # cn(u,v) = (A^2)[u,v]
        A2 = adj @ adj
        # Sum of C(cn,2) over all unordered pairs
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                cn = A2[u, v]
                if cn >= 2:
                    total += cn * (cn - 1) // 2
        return total // 2
    
    # Find best edge to add (minimizes new C4s)
    A2 = adj @ adj
    min_cn = float('inf')
    best_edge = None
    
    for u in range(n):
        for v in range(u+1, n):
            if not adj[u, v]:
                cn = A2[u, v]
                if cn < min_cn:
                    min_cn = cn
                    best_edge = (u, v)
    
    if best_edge is None:
        # Fallback: return first 91 edges
        edge_list = []
        for (u, v) in sorted(edges_set):
            edge_list.append([u, v])
            if len(edge_list) == m:
                break
        return edge_list
    
    # Add the best edge
    u, v = best_edge
    adj[u, v] = 1
    adj[v, u] = 1
    edges_set.add(best_edge)
    
    # Local search: try swaps to reduce C4 count
    current_c4 = count_c4(adj)
    
    # Precompute common neighbors for all pairs
    def get_common_neighbors(u, v, adj):
        return np.sum(adj[u] & adj[v])
    
    # Try all possible swaps: remove edge (a,b), add edge (c,d)
    # where (a,b) is an existing edge and (c,d) is a non-edge
    # The change in C4 count can be computed incrementally
    
    # For efficiency, precompute adjacency lists
    def adj_list(adj):
        return [list(np.where(adj[i] == 1)[0]) for i in range(n)]
    
    # Local search iterations
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        best_swap = None
        best_c4 = current_c4
        
        # Get current edges and non-edges
        current_edges = list(edges_set)
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u, v]:
                    non_edges.append((u, v))
        
        # Try each swap
        for (a, b) in current_edges:
            if time.time() > deadline - 10:
                break
            # Remove edge (a,b)
            adj[a, b] = 0
            adj[b, a] = 0
            
            # Compute new A2 for common neighbors
            # Instead of full matrix mult, compute incrementally
            # Removing edge (a,b) affects A2[a,*], A2[*,a], A2[b,*], A2[*,b]
            
            # For each non-edge (c,d), compute new C4 count
            for (c, d) in non_edges:
                if c == a or c == b or d == a or d == b:
                    continue  # Can't add edge involving a or b if we just removed (a,b)... actually we can
                # Actually we can add any non-edge, even if it involves a or b
                
                # Compute change in C4 count
                # This is complex; let's just compute directly for small n
                adj[c, d] = 1
                adj[d, c] = 1
                
                new_c4 = count_c4(adj)
                
                adj[c, d] = 0
                adj[d, c] = 0
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_swap = ((a, b), (c, d))
            
            # Restore edge (a,b)
            adj[a, b] = 1
            adj[b, a] = 1
        
        if best_swap is not None:
            (a, b), (c, d) = best_swap
            # Apply swap
            adj[a, b] = 0
            adj[b, a] = 0
            edges_set.remove((a, b))
            adj[c, d] = 1
            adj[d, c] = 1
            edges_set.add((c, d))
            current_c4 = best_c4
            improved = True
    
    # Return edges
    edge_list = []
    for (u, v) in sorted(edges_set):
        edge_list.append([u, v])
    
    return edge_list
