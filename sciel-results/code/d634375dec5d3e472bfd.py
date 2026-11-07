"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
    
    # Build adjacency for base graph
    adj_base = [[] for _ in range(n)]
    for u, v in base_edges:
        adj_base[u].append(v)
        adj_base[v].append(u)
    
    # Candidate edges: all non-adjacent pairs that don't form triangles with existing edges
    # An edge (u,v) is valid if:
    # 1. (u,v) not in base_edges
    # 2. u and v have no common neighbor in the base graph (triangle-free constraint)
    #    Actually, we need to be more careful: adding (u,v) creates a triangle if there exists w
    #    such that (u,w) and (v,w) are both in the graph. Since we're adding edges one by one,
    #    we need to ensure no triangle is formed. But for the set cover approach, we need to
    #    consider which edges can be added without creating triangles.
    
    # For triangle-free: (u,v) can be added only if N(u) ∩ N(v) = ∅ in the current graph.
    # But since we're selecting a set of edges simultaneously, we need: for any two added edges
    # (u1,v1) and (u2,v2), they shouldn't create triangles. This is complex.
    
    # Simpler approach: pre-compute all candidate edges that are "safe" individually,
    # then use ILP or greedy with verification.
    
    # Let's first identify all candidate edges that don't form a triangle with base edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges:
                continue
            # Check if u and v share a common neighbor
            common = set(adj_base[u]) & set(adj_base[v])
            if len(common) == 0:
                candidates.append((u, v))
    
    # Now we need to select a subset of candidates such that:
    # 1. The resulting graph is triangle-free
    # 2. The resulting graph is connected
    # 3. The diameter is at most 4
    # 4. Minimize the number of added edges
    
    # For triangle-free with multiple added edges: we need to ensure that no three vertices
    # form a triangle in the final graph. This means for any added edge (u,v), no other added
    # edge (u,w) or (v,w) can exist where (w, the other) is also present... 
    # Actually, a triangle consists of 3 edges. So we need: for any triple (a,b,c), not all
    # three edges (a,b), (b,c), (a,c) are present (either in base or added).
    
    # Let's use a greedy approach with verification, trying to minimize edges.
    
    def bfs_distances(adj):
        """Compute all-pairs shortest paths using BFS from each node."""
        dist = [[-1] * n for _ in range(n)]
        for start in range(n):
            dist[start][start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if dist[start][neighbor] == -1:
                        dist[start][neighbor] = dist[start][node] + 1
                        queue.append(neighbor)
        return dist
    
    def get_diameter(dist):
        """Get the diameter from distance matrix."""
        max_d = 0
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i][j] == -1:
                    return -1  # disconnected
                max_d = max(max_d, dist[i][j])
        return max_d
    
    def is_triangle_free(adj):
        """Check if graph is triangle-free."""
        for u in range(n):
            for i, v in enumerate(adj[u]):
                if v <= u:
                    continue
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False
        return True
    
    def check_validity(added_edges):
        """Check if adding these edges to base graph satisfies all constraints."""
        adj = [list(neighbors) for neighbors in adj_base]
        edge_set = set(base_edges)
        
        for u, v in added_edges:
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
            edge_set.add((v, u))
            adj[u].append(v)
            adj[v].append(u)
        
        # Check triangle-free
        if not is_triangle_free(adj):
            return False
        
        # Check connectivity and diameter
        dist = bfs_distances(adj)
        diam = get_diameter(dist)
        if diam == -1 or diam > 4:
            return False
        
        return True
    
    # Try to find minimum number of edges
    # Start with 0 edges and increase
    
    # First, check if base graph already satisfies (it doesn't, diameter is 23)
    # Base graph is a path, so diameter is 23.
    
    # Let's try small numbers of added edges
    # For efficiency, let's use a greedy approach first, then try to improve
    
    # Greedy: iteratively add the edge that most reduces the diameter
    # while maintaining triangle-free and connectivity
    
    def greedy_solve():
        adj = [list(neighbors) for neighbors in adj_base]
        edge_set = set(base_edges)
        added = []
        
        # Compute current diameter
        dist = bfs_distances(adj)
        current_diam = get_diameter(dist)
        
        while current_diam > 4:
            best_edge = None
            best_diam = current_diam + 1  # infinity-like
            
            for u, v in candidates:
                if (u, v) in edge_set:
                    continue
                
                # Tentatively add edge
                new_adj = [list(neighbors) for neighbors in adj]
                new_adj[u].append(v)
                new_adj[v].append(u)
                
                # Check triangle-free
                if not is_triangle_free(new_adj):
                    continue
                
                # Check connectivity and diameter
                new_dist = bfs_distances(new_adj)
                new_diam = get_diameter(new_dist)
                if new_diam == -1 or new_diam > 4:
                    continue
                
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_edge = (u, v)
            
            if best_edge is None:
                return None  # stuck
            
            u, v = best_edge
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((u, v))
            edge_set.add((v, u))
            added.append([u, v])
            current_diam = best_diam
        
        return added
    
    result = greedy_solve()
    
    if result is not None:
        # Verify the result
        if check_validity(result):
            # Convert to proper types
            return [[int(u), int(v)] for u, v in result]
    
    # Fallback: return empty list (shouldn't happen)
    return []

# EVOLVE-BLOCK-END
