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
    
    def get_diameter_and_farthest():
        max_diam = 0
        far_pairs = []
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
            if -1 in dist:
                return float('inf'), []
            local_max = max(dist)
            if local_max > max_diam:
                max_diam = local_max
                far_pairs = [(start, end) for end in range(n) if dist[end] == local_max]
            elif local_max == max_diam:
                far_pairs.extend([(start, end) for end in range(n) if dist[end] == local_max])
        return max_diam, far_pairs
    
    def is_valid(u, v):
        if u == v:
            return False
        u, v = min(u,v), max(u,v)
        if (u,v) in existing_edges or (u,v) in added_set:
            return False
        if adj[u] & adj[v]:
            return False
        return True
    
    def add_edge(u, v):
        u, v = min(u,v), max(u,v)
        adj[u].add(v)
        adj[v].add(u)
        added_edges.append([int(u), int(v)])
        added_set.add((u,v))
    
    deadline = time.time() + 55.0
    
    while time.time() < deadline:
        current_diam, far_pairs = get_diameter_and_farthest()
        if current_diam <= 4:
            break
        
        best_edge = None
        best_new_diam = current_diam + 1
        best_far_reduction = 0
        
        # Precompute distances for efficiency
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
        
        for u in range(n):
            for v in range(u + 1, n):
                if not is_valid(u, v):
                    continue
                
                # New distance between any pair (a,b) with edge (u,v):
                # min(old_dist(a,b), dist(a,u)+1+dist(v,b), dist(a,v)+1+dist(u,b))
                # For diameter, we need max over all pairs
                # Approximate: check the far pairs
                
                new_diam = current_diam
                for a, b in far_pairs:
                    da_u = all_dists[a][u]
                    da_v = all_dists[a][v]
                    db_u = all_dists[b][u]
                    db_v = all_dists[b][v]
                    if da_u == -1 or da_v == -1 or db_u == -1 or db_v == -1:
                        new_dist_ab = all_dists[a][b]
                    else:
                        new_dist_ab = min(all_dists[a][b], da_u + 1 + db_v, da_v + 1 + db_u)
                    new_diam = min(new_diam, new_dist_ab)
                
                # This is an underestimate of new diameter. For accuracy, do full BFS if promising.
                if new_diam < best_new_diam:
                    # Verify with full BFS
                    adj[u].add(v)
                    adj[v].add(u)
                    added_set.add((min(u,v), max(u,v)))
                    
                    real_new_diam, _ = get_diameter_and_farthest()
                    
                    adj[u].remove(v)
                    adj[v].remove(u)
                    added_set.remove((min(u,v), max(u,v)))
                    
                    if real_new_diam < best_new_diam:
                        best_new_diam = real_new_diam
                        best_edge = (u, v)
                        best_far_reduction = current_diam - real_new_diam
                
                if best_new_diam <= 4:
                    break
            
            if best_new_diam <= 4:
                break
        
        if best_edge is None:
            break
        
        add_edge(best_edge[0], best_edge[1])
    
    return [[int(u), int(v)] for u, v in added_edges]
# EVOLVE-BLOCK-END
