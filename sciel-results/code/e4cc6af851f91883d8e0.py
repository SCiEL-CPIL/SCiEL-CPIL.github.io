"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build adjacency for base graph
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute base distances
    def compute_distances(adj, n):
        dist = [[-1] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[i][v] == -1:
                        dist[i][v] = dist[i][u] + 1
                        q.append(v)
        return dist
    
    def get_diameter(adj, n):
        dist = compute_distances(adj, n)
        max_d = 0
        for i in range(n):
            for j in range(n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
        return max_d
    
    def is_triangle_free(adj, n):
        for i in range(n):
            neighbors = list(adj[i])
            for a_idx in range(len(neighbors)):
                for b_idx in range(a_idx + 1, len(neighbors)):
                    a = neighbors[a_idx]
                    b = neighbors[b_idx]
                    if b in adj[a]:
                        return False
        return True
    
    # Candidate edges: all non-adjacent pairs
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                candidates.append((u, v))
    
    # Greedy approach: repeatedly pick the edge that reduces the most "excess" distance
    # while maintaining triangle-free property
    
    def try_add_edge(adj, u, v, n):
        """Try to add edge (u,v), return new adj or None if triangle created"""
        # Check if adding (u,v) creates a triangle
        for w in adj[u]:
            if w in adj[v]:
                return None
        new_adj = [set(s) for s in adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        return new_adj
    
    # Start with base graph
    current_adj = [set(s) for s in adj]
    added_edges = []
    
    # Compute which pairs are too far
    def get_far_pairs(adj, n, max_dist=4):
        dist = compute_distances(adj, n)
        far = []
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i][j] > max_dist:
                    far.append((i, j, dist[i][j]))
        return far
    
    # Greedy: pick edge that covers most far pairs
    while time.time() < deadline:
        far_pairs = get_far_pairs(current_adj, n)
        if not far_pairs:
            break
        
        # For each candidate edge, compute how many far pairs it would "help" with
        # An edge (u,v) helps pair (a,b) if it reduces dist(a,b)
        best_edge = None
        best_score = -1
        
        # Only consider edges that could help
        for u, v in candidates:
            if (u, v) in base_edges:
                continue
            if v in current_adj[u]:
                continue
            
            # Check triangle-free
            new_adj = try_add_edge(current_adj, u, v, n)
            if new_adj is None:
                continue
            
            # Compute how many far pairs this edge helps
            # Simple heuristic: count far pairs where u or v is an endpoint, 
            # or where the edge creates a shorter path
            score = 0
            dist_new = compute_distances(new_adj, n)
            for a, b, d in far_pairs:
                if dist_new[a][b] < d:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_edge = (u, v)
                best_new_adj = new_adj
            
            if time.time() > deadline:
                break
        
        if best_edge is None:
            break
        
        u, v = best_edge
        current_adj = best_new_adj
        added_edges.append((u, v))
        base_edges.add((u, v))
    
    # Verify final result
    final_diam = get_diameter(current_adj, n)
    if final_diam > 4 or not is_triangle_free(current_adj, n):
        # Fallback: return empty or a known valid solution
        # Try a simple valid solution
        pass
    
    # Convert to list of lists
    result = [[u, v] for u, v in added_edges]
    
    # Ensure types are correct
    final_result = []
    for e in result:
        final_result.append([int(e[0]), int(e[1])])
    
    return final_result
# EVOLVE-BLOCK-END
