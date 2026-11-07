import numpy as np
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    # Initialize distance matrix
    dist = np.full((n, n), 9999)
    for i in range(n):
        dist[i][i] = 0
    for u, v in P['edges']:
        dist[u][v] = 1
        dist[v][u] = 1
    
    # Initial Floyd-Warshall to get correct distances (path is simple, but good practice)
    for k in range(n):
        dist = np.minimum(dist, dist[:, k][:, None] + dist[k, :])
        
    added_edges = []
    added_set = set()
    
    # Helper to check if adding (u, v) creates a triangle
    def is_valid_add(u, v):
        # Check against existing added edges
        # Triangle (u, v, w) where (u, w) and (v, w) are in added_set
        # Or (u, v) and (u, w) and (v, w)
        # Actually, just check if there exists w such that {u,w} and {v,w} are in added_set
        # Or if {u,w} in added_set and |v-w|==1 (path edge)
        # Or if {v,w} in added_set and |u-w|==1 (path edge)
        
        # Iterate through added edges to find conflicts
        for a, b in added_set:
            # Case 1: (u, v) and (a, b) share a vertex and the third vertex is connected by path or added edge
            # If a == u or b == u:
            if a == u or b == u:
                w = b if a == u else a
                # We have edge (u, w) in added_set.
                # Adding (u, v).
                # Triangle if (w, v) is an edge (path or added).
                if abs(w - v) == 1:
                    return False
                if (min(w, v), max(w, v)) in added_set:
                    return False
            if a == v or b == v:
                w = b if a == v else a
                # We have edge (v, w) in added_set.
                # Adding (u, v).
                # Triangle if (w, u) is an edge (path or added).
                if abs(w - u) == 1:
                    return False
                if (min(w, u), max(w, u)) in added_set:
                    return False
            
            # Case 2: (u, v) and (a, b) are disjoint?
            # No triangle formed directly.
            # But what if (u, v) and (a, b) and (u, a)?
            # This is covered by Case 1 if we iterate all pairs?
            # Let's re-verify.
            # A triangle involves 3 vertices.
            # If we add (u, v), the triangle must include (u, v).
            # So the other two edges must connect to u or v.
            # So the other edges must be (u, w) and (v, w) for some w.
            # So we just need to check if there exists w such that (u, w) and (v, w) are in the graph.
            # The graph contains path edges and added edges.
            # Path edges: (i, i+1).
            # Added edges: added_set.
            
            # So, for a fixed (u, v):
            # Check if there is w such that (u, w) is path/added AND (v, w) is path/added.
            # If (u, w) is path, w is u-1 or u+1.
            # If (v, w) is path, w is v-1 or v+1.
            
            # Let's just iterate w from 0 to n-1.
            # This is O(N) per check.
            for w in range(n):
                if w == u or w == v: continue
                edge_uw = (min(u, w), max(u, w))
                edge_vw = (min(v, w), max(v, w))
                
                is_uw_edge = (abs(u-w) == 1) or (edge_uw in added_set)
                is_vw_edge = (abs(v-w) == 1) or (edge_vw in added_set)
                
                if is_uw_edge and is_vw_edge:
                    return False
        return True

    start_time = time.time()
    deadline = start_time + 50 # Safety margin
    
    while True:
        if time.time() > deadline:
            break
            
        current_diam = int(np.max(dist))
        if current_diam <= 4:
            break
            
        best_edge = None
        best_new_diam = current_diam + 1 # Must be better or equal? We want to minimize.
        
        # Generate candidates
        for u in range(n):
            for v in range(u + 3, n): # v-u >= 3 to avoid triangles with path (dist 2)
                if (u, v) in added_set:
                    continue
                if not is_valid_add(u, v):
                    continue
                
                # Simulate
                new_dist = dist.copy()
                new_dist[u, v] = 1
                new_dist[v, u] = 1
                
                # Update distances
                # Using NumPy Floyd-Warshall
                for k in range(n):
                    new_dist = np.minimum(new_dist, new_dist[:, k][:, None] + new_dist[k, :])
                
                new_diam = int(np.max(new_dist))
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
                    
        if best_edge is None:
            # If no edge improves diameter, we might be stuck or done?
            # If current_diam > 4 and no edge reduces it, we fail?
            # But we should always be able to reduce diameter by adding edges.
            # Maybe the greedy choice is bad.
            # For now, break to avoid infinite loop.
            break
            
        # Add the edge
        u, v = best_edge
        added_set.add((u, v))
        added_edges.append([u, v])
        dist[u, v] = 1
        dist[v, u] = 1
        for k in range(n):
            dist = np.minimum(dist, dist[:, k][:, None] + dist[k, :])
            
    # Return added edges
    return [[int(u), int(v)] for u, v in added_edges]
