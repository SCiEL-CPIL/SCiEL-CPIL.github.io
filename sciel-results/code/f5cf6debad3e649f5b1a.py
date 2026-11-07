import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    # Strategy: Use TWO hubs to reduce the number of edges needed or improve diameter robustness.
    # Hub 1: Vertex 6 (even). Connects to odd vertices in the left half.
    # Hub 2: Vertex 17 (odd). Connects to even vertices in the right half.
    # 
    # Let's analyze the distances.
    # Path: 0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23
    #
    # If we add edges from Hub 6 (even) to odd vertices {1, 3, 5, 9, 11, 13, 15}
    # And edges from Hub 17 (odd) to even vertices {2, 4, 10, 12, 14, 16, 18, 20, 22}
    #
    # Wait, we need to be careful about triangles.
    # Hub 6 is even. Neighbors in path: 5, 7.
    # If we connect 6 to 1: Path 6-5-4-3-2-1. No triangle.
    # If we connect 6 to 3: Path 6-5-4-3. No triangle.
    # If we connect 6 to 5: Already edge.
    # If we connect 6 to 7: Already edge.
    # If we connect 6 to 9: Path 6-7-8-9. No triangle.
    #
    # Let's try a specific construction that minimizes edges.
    # The current single hub (11) added 10 edges. Score 12.1.
    # Can we do better with 2 hubs?
    
    # Let's try Hub 5 (odd) and Hub 18 (even).
    # Hub 5 connects to even vertices in [0, 11]: 0, 2, 4, 8, 10. (Exclude 4, 6 which are neighbors? No, 4 and 6 are neighbors. Exclude them to avoid duplicate edges, but they are already connected. We want to shorten distances.)
    # Actually, connecting to neighbors is useless.
    # Hub 5 (odd) -> Even vertices {0, 2, 8, 10}. (4 is neighbor, 6 is neighbor).
    # Hub 18 (even) -> Odd vertices {7, 9, 13, 15, 17, 21, 23}. (17, 19 are neighbors).
    
    # Let's check diameter for this set.
    # Added edges:
    # (5, 0), (5, 2), (5, 8), (5, 10)
    # (18, 7), (18, 9), (18, 13), (18, 15), (18, 17), (18, 21), (18, 23)
    # Total 11 edges. Worse than 10.
    
    # Let's try to optimize the single hub approach first? No, directive says structural change.
    # But maybe the single hub wasn't optimal.
    # Hub 11 (odd).
    # Even vertices: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22.
    # Neighbors of 11: 10, 12.
    # Distances from 11 in path:
    # 11->10 (1), 11->12 (1)
    # 11->9 (2), 11->13 (2)
    # 11->8 (3), 11->14 (3)
    # 11->7 (4), 11->15 (4)
    # 11->6 (5), 11->16 (5)
    # 11->5 (6), 11->17 (6)
    # 11->4 (7), 11->18 (7)
    # 11->3 (8), 11->19 (8)
    # 11->2 (9), 11->20 (9)
    # 11->1 (10), 11->21 (10)
    # 11->0 (11), 11->22 (11)
    # 11->23 (12)
    
    # If we add edge (11, v), dist(11, v) becomes 1.
    # We need max distance in graph <= 4.
    
    # Let's use a BFS-based greedy search to find the minimal set of edges.
    # This is a different *method* (search/optimization) rather than a fixed heuristic.
    
    import itertools
    
    # Precompute base distances
    def get_base_dist(src, n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        # Build base adjacency
        base_adj = {i: [] for i in range(n)}
        for u, v in base_edges:
            base_adj[u].append(v)
            base_adj[v].append(u)
            
        while q:
            u = q.popleft()
            for v in base_adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist
    
    base_dists = [get_base_dist(i, n) for i in range(n)]
    
    # Candidate edges: All non-existing edges that don't create triangles.
    # An edge (u, v) creates a triangle if there exists w such that (u, w) and (v, w) are in the graph.
    # Since the base graph is a path, the only triangles can be formed if u and v are both connected to the same neighbor in the path?
    # No, in a path, neighbors are unique.
    # If we add edge (u, v), a triangle is formed if there is a vertex w such that w is adjacent to u and w is adjacent to v in the *current* graph.
    # Initially, current graph is path.
    # If we add (u, v), and u, v are not adjacent in path, no triangle is formed immediately unless we add more edges.
    # However, we must ensure the *final* graph is triangle-free.
    
    # Let's define a function to check if adding edge (u, v) to a set of added edges creates a triangle.
    def is_triangle_free(base_set, added_edges, u, v):
        # Check if u and v have a common neighbor in the current graph (base + added)
        # We can just check against base neighbors and added neighbors.
        
        # Build adjacency set for quick lookup? Or just check pairs.
        # For small n=24, we can check all w.
        for w in range(n):
            if w == u or w == v:
                continue
            # Check if (u, w) exists
            edge_uw = (min(u, w), max(u, w))
            edge_vw = (min(v, w), max(v, w))
            
            # u-w exists if in base or added
            if (edge_uw in base_set or edge_uw in added_edges_set) and (edge_vw in base_set or edge_vw in added_edges_set):
                return False
        return True
    
    # We will use a greedy set cover approach.
    # Goal: Minimize number of edges such that diameter <= 4.
    
    # First, identify "critical" pairs (u, v) that are too far apart in the base graph.
    # If dist_base(u, v) > 4, we need to shorten their path.
    # Actually, we just need the *resulting* diameter to be <= 4.
    
    # Let's try a simple heuristic:
    # 1. Start with empty added edges.
    # 2. Find pair (u, v) with max distance in current graph.
    # 3. If dist > 4, find an edge (x, y) that reduces this distance most effectively, without creating triangles.
    # 4. Add it.
    # 5. Repeat.
    
    # This is a "Greedy Diameter Reduction" strategy.
    
    added_edges = []
    added_edges_set = set()
    
    def get_current_diameter_and_worst_pair(adj):
        # Returns (diameter, (u, v))
        max_d = 0
        worst = (-1, -1)
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for d in dist:
                if d > max_d:
                    max_d = d
                    # We need to find the pair. We can store it during BFS or re-run.
                    # For efficiency, let's just return max_d and find pair in a second pass if needed.
                    # Or just return max_d.
            # To find worst pair, we can track it.
        # This is O(N^2) per check. N=24, so 576 BFS. Very fast.
        
        # Let's do it properly to get the pair.
        max_d = 0
        worst_u, worst_v = -1, -1
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for dst in range(n):
                if dist[dst] > max_d:
                    max_d = dist[dst]
                    worst_u = src
                    worst_v = dst
        return max_d, (worst_u, worst_v)

    # Build initial adjacency
    adj = {i: set() for i in range(n)}
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    while True:
        diam, (u, v) = get_current_diameter_and_worst_pair(adj)
        if diam <= 4:
            break
            
        # We need to add an edge to reduce the distance between u and v (or generally reduce diameter).
        # Candidate edges: (x, y) not in base, not in added.
        # To be effective, it should likely be "short" in terms of graph distance? Or connect hubs?
        # Let's try all possible candidate edges and pick the one that minimizes the new diameter.
        
        best_edge = None
        best_new_diam = diam + 1
        
        # Generate candidates
        for x in range(n):
            for y in range(x + 1, n):
                edge = (min(x, y), max(x, y))
                if edge in base_set or edge in added_edges_set:
                    continue
                
                # Check triangle-free
                if not is_triangle_free(base_set, added_edges, x, y):
                    continue
                
                # Tentatively add edge
                adj[x].add(y)
                adj[y].add(x)
                
                new_diam, _ = get_current_diameter_and_worst_pair(adj)
                
                # Remove edge
                adj[x].remove(y)
                adj[y].remove(x)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (x, y)
                    
        if best_edge is None:
            # No valid edge reduces diameter? This shouldn't happen if graph is connected and not a path.
            # But if it does, we are stuck.
            break
            
        x, y = best_edge
        added_edges.append([x, y])
        added_edges_set.add((min(x, y), max(x, y)))
        adj[x].add(y)
        adj[y].add(x)
        
        # Safety check: if we add too many edges, break
        if len(added_edges) > 15:
            break

    # Final verification
    # Check Triangle Free
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[u]:
                    if w != v and w in adj[v]:
                        return [] # Invalid, fallback (should not happen)
                        
    # Check Diameter
    diam, _ = get_current_diameter_and_worst_pair(adj)
    if diam > 4:
        return []
        
    return [[int(u), int(v)] for u, v in added_edges]

# EVOLVE-BLOCK-END
