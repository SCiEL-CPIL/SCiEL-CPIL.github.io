"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check if adding edge (u,v) would create a triangle
    def creates_triangle(u, v, adj):
        # Check if any common neighbor exists
        for w in adj[u]:
            if w == v:
                continue
            if w in adj[v]:
                return True
        return False
    
    # BFS to compute all-pairs distances
    def all_pairs_distances(adj):
        n = len(adj)
        dist = [[-1] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            queue = deque([i])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[i][v] == -1:
                        dist[i][v] = dist[i][u] + 1
                        queue.append(v)
        return dist
    
    def max_distance(adj):
        dist = all_pairs_distances(adj)
        mx = 0
        for i in range(n):
            for j in range(n):
                if dist[i][j] > mx:
                    mx = dist[i][j]
        return mx
    
    def uncovered_pairs(adj):
        """Return set of pairs (i,j) with i<j and dist > 4"""
        dist = all_pairs_distances(adj)
        uncovered = set()
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] > 4:
                    uncovered.add((i, j))
        return uncovered
    
    # Greedy set cover approach
    current_adj = [set(s) for s in adj]
    added_edges = []
    start_time = time.time()
    
    while True:
        if time.time() - start_time > 50:
            break
        
        max_d = max_distance(current_adj)
        if max_d <= 4:
            break
        
        uncovered = uncovered_pairs(current_adj)
        if not uncovered:
            break
        
        # Find the best edge to add
        best_edge = None
        best_count = -1
        
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) in base_edges or (v, u) in base_edges:
                    continue
                # Check if edge is already added
                if any((u, e[0]) == (e[1], e[1]) or (u, e[1]) == (e[0], e[1]) for e in added_edges):
                    continue
                # Check triangle-freeness
                if creates_triangle(u, v, current_adj):
                    continue
                
                # Count how many uncovered pairs this edge would cover
                # An edge (u,v) helps pairs (a,b) where a is near u and b is near v (or vice versa)
                # Specifically, it reduces dist(a,b) if a is within some range of u and b within some range of v
                # Let's be more precise: after adding (u,v), dist(a,b) <= dist(a,u)+1+dist(v,b) or dist(a,v)+1+dist(u,b)
                # We need to check which uncovered pairs get fixed
                
                # Compute distances from u and v in current graph
                dist_u = [-1] * n
                dist_u[u] = 0
                queue = deque([u])
                while queue:
                    x = queue.popleft()
                    for y in current_adj[x]:
                        if dist_u[y] == -1:
                            dist_u[y] = dist_u[x] + 1
                            queue.append(y)
                
                dist_v = [-1] * n
                dist_v[v] = 0
                queue = deque([v])
                while queue:
                    x = queue.popleft()
                    for y in current_adj[x]:
                        if dist_v[y] == -1:
                            dist_v[y] = dist_v[x] + 1
                            queue.append(y)
                
                count = 0
                for (a, b) in uncovered:
                    # New distance via u-v edge: min(dist(a,u)+1+dist(v,b), dist(a,v)+1+dist(u,b))
                    new_d1 = dist_u[a] + 1 + dist_v[b] if dist_u[a] != -1 and dist_v[b] != -1 else float('inf')
                    new_d2 = dist_u[b] + 1 + dist_v[a] if dist_u[b] != -1 and dist_v[a] != -1 else float('inf')
                    new_dist = min(new_d1, new_d2)
                    if new_dist <= 4:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_edge = (u, v)
        
        if best_edge is None or best_count <= 0:
            break
        
        u, v = best_edge
        added_edges.append([u, v])
        current_adj[u].add(v)
        current_adj[v].add(u)
    
    # Convert to proper int lists
    result = [[int(u), int(v)] for u, v in added_edges]
    return result

# EVOLVE-BLOCK-END
