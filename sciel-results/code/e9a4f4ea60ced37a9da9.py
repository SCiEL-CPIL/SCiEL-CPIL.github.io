import time
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    existing_edges = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        existing_edges.add((min(u,v), max(u,v)))
    
    added_edges = []
    added_set = set()
    
    def compute_all_dists():
        """Compute all-pairs shortest paths via BFS from each node."""
        all_dists = []
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            all_dists.append(dist)
        return all_dists
    
    def get_diameter(all_dists):
        max_d = 0
        for i in range(n):
            for j in range(n):
                if all_dists[i][j] > max_d:
                    max_d = all_dists[i][j]
        return max_d
    
    def is_valid(u, v):
        if u == v:
            return False
        a, b = min(u,v), max(u,v)
        if (a,b) in existing_edges or (a,b) in added_set:
            return False
        if adj[u] & adj[v]:
            return False
        return True
    
    def add_edge(u, v):
        a, b = min(u,v), max(u,v)
        adj[u].add(v)
        adj[v].add(u)
        added_edges.append([int(a), int(b)])
        added_set.add((a,b))
    
    deadline = time.time() + 55.0
    
    while time.time() < deadline:
        all_dists = compute_all_dists()
        current_diam = get_diameter(all_dists)
        if current_diam <= 4:
            break
        
        # Find all pairs at maximum distance
        far_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                if all_dists[i][j] == current_diam:
                    far_pairs.append((i, j))
        
        best_edge = None
        best_new_diam = current_diam + 1
        best_far_reduction = 0
        
        # Try all valid edges
        for u in range(n):
            for v in range(u + 1, n):
                if not is_valid(u, v):
                    continue
                
                # Quick estimate: does this edge help any far pair?
                # New dist between (a,b) via (u,v): min(old, dist(a,u)+1+dist(v,b), dist(a,v)+1+dist(u,b))
                helps = False
                for a, b in far_pairs:
                    da_u = all_dists[a][u]
                    da_v = all_dists[a][v]
                    db_u = all_dists[b][u]
                    db_v = all_dists[b][v]
                    if da_u == -1 or da_v == -1 or db_u == -1 or db_v == -1:
                        continue
                    new_ab = min(all_dists[a][b], da_u + 1 + db_v, da_v + 1 + db_u)
                    if new_ab < all_dists[a][b]:
                        helps = True
                        break
                
                if not helps:
                    continue
                
                # Full verification
                adj[u].add(v)
                adj[v].add(u)
                
                new_all_dists = compute_all_dists()
                new_diam = get_diameter(new_all_dists)
                
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
                    best_far_reduction = current_diam - new_diam
                elif new_diam == best_new_diam and best_edge is not None:
                    # Tie-break: prefer edges that reduce more far pairs
                    pass  # Keep first found for simplicity
            
            if best_new_diam <= 4:
                break
        
        if best_edge is None:
            break
        
        add_edge(best_edge[0], best_edge[1])
    
    return [[int(u), int(v)] for u, v in added_edges]
# EVOLVE-BLOCK-END
