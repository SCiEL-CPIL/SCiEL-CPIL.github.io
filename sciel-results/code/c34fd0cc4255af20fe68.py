"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import heapq
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    
    # State
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u, v), max(u, v)))
        
    added_edges = []

    def get_diameter():
        max_d = 0
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                node = q.popleft()
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        q.append(nei)
                        if dist[nei] > max_d:
                            max_d = dist[nei]
        return max_d

    def is_triangle_free():
        # Check all triangles
        # For each edge (u,v), check if they share a neighbor
        # Optimized: For each node u, check pairs of neighbors
        for u in range(n):
            neighbors = adj[u]
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        return True

    def would_create_triangle(u, v):
        # Check if adding (u,v) creates a triangle
        # Common neighbor?
        for w in adj[u]:
            if w in adj[v]:
                return True
        return False

    def try_add_edge(u, v):
        if (u, v) in edge_set:
            return False
        if u == v:
            return False
        if would_create_triangle(u, v):
            return False
        
        # Temporarily add
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u, v), max(u, v)))
        added_edges.append([u, v])
        
        # Check if this helps (we assume it does in the add phase, 
        # but we verify global validity in the main loop)
        return True

    def try_remove_edge(u, v):
        # Temporarily remove
        adj[u].remove(v)
        adj[v].remove(u)
        edge_set.remove((min(u, v), max(u, v)))
        added_edges.remove([u, v])
        
        # Check validity
        if not is_triangle_free():
            # Restore
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u, v), max(u, v)))
            added_edges.append([u, v])
            return False
            
        d = get_diameter()
        if d > 4:
            # Restore
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u, v), max(u, v)))
            added_edges.append([u, v])
            return False
            
        return True

    # Phase 1: Greedy Add
    # We add edges until diameter <= 4 and triangle-free
    # To ensure triangle-free, we only add edges that don't create triangles.
    # We pick the edge that reduces the diameter the most.
    
    while True:
        d = get_diameter()
        if d <= 4:
            break
            
        # Find best edge to add
        best_score = -1
        best_edge = None
        
        # Precompute distances for scoring
        dists = []
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                node = q.popleft()
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        q.append(nei)
            dists.append(dist)
            
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) in edge_set:
                    continue
                if would_create_triangle(u, v):
                    continue
                
                # Score: Reduction in diameter or max distance
                # New diameter is at most max(old_d, max over pairs of new_dist)
                # We approximate by checking the worst pair (0, n-1) or similar?
                # Better: Count how many pairs have distance > 4 and are reduced.
                
                # If we add (u,v), new dist between i,j is min(old, dist[i,u]+1+dist[v,j], dist[i,v]+1+dist[u,j])
                # We want to minimize the number of pairs with dist > 4.
                current_bad = 0
                new_bad = 0
                for i in range(n):
                    for j in range(i+1, n):
                        old_d = dists[i][j]
                        if old_d > 4:
                            current_bad += 1
                            new_d = min(old_d, 
                                        dists[i][u] + 1 + dists[v][j],
                                        dists[i][v] + 1 + dists[u][j])
                            if new_d > 4:
                                new_bad += 1
                            # Also check if new_d is -1 (shouldn't happen if connected)
                
                improvement = current_bad - new_bad
                # Tie breaker: reduce max distance
                max_old = max(max(row) for row in dists)
                max_new = max_old
                # This is expensive to compute exactly for every edge.
                # Use improvement as primary score.
                
                if improvement > best_score:
                    best_score = improvement
                    best_edge = (u, v)
                    
        if best_edge is None:
            # If no edge can be added without creating a triangle, 
            # but diameter is still > 4, we are stuck.
            # This shouldn't happen for this specific graph if we are careful,
            # but if it does, we break to avoid infinite loop.
            break
            
        u, v = best_edge
        try_add_edge(u, v)

    # Phase 2: Greedy Remove
    # Try to remove edges one by one
    changed = True
    while changed:
        changed = False
        # Iterate over a copy of added edges to allow removal
        for edge in list(added_edges):
            u, v = edge
            if try_remove_edge(u, v):
                changed = True
                
    # Phase 3: Local Search / Swaps (Optional but good for optimality)
    # Try swapping one edge for another if it allows further reduction?
    # Or just repeat Add/Remove a few times.
    
    # Repeat Add/Remove a few times to escape local minima
    for _ in range(5):
        # Add phase
        while True:
            d = get_diameter()
            if d <= 4:
                break
            best_score = -1
            best_edge = None
            dists = []
            for i in range(n):
                dist = [-1] * n
                dist[i] = 0
                q = deque([i])
                while q:
                    node = q.popleft()
                    for nei in adj[node]:
                        if dist[nei] == -1:
                            dist[nei] = dist[node] + 1
                            q.append(nei)
                dists.append(dist)
            
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in edge_set:
                        continue
                    if would_create_triangle(u, v):
                        continue
                    current_bad = 0
                    new_bad = 0
                    for i in range(n):
                        for j in range(i+1, n):
                            old_d = dists[i][j]
                            if old_d > 4:
                                current_bad += 1
                                new_d = min(old_d, 
                                            dists[i][u] + 1 + dists[v][j],
                                            dists[i][v] + 1 + dists[u][j])
                                if new_d > 4:
                                    new_bad += 1
                    improvement = current_bad - new_bad
                    if improvement > best_score:
                        best_score = improvement
                        best_edge = (u, v)
                        
            if best_edge is None:
                break
            u, v = best_edge
            try_add_edge(u, v)
            
        # Remove phase
        changed = True
        while changed:
            changed = False
            for edge in list(added_edges):
                u, v = edge
                if try_remove_edge(u, v):
                    changed = True

    return added_edges
# EVOLVE-BLOCK-END
