import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        if u > v:
            u, v = v, u
        base_edges.add((u, v))
    
    # Generate valid candidates: opposite parity, not in base
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u + v) % 2 == 1 and (u, v) not in base_edges:
                candidates.append((u, v))
    
    def build_adj(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def bfs_diameter(adj):
        diam = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            maxd = 0
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > maxd:
                            maxd = dist[nb]
                        q.append(nb)
            if maxd > diam:
                diam = maxd
        return diam
    
    def get_farthest_pair(adj):
        """Return (diam, (s, t)) where dist(s,t) == diam."""
        best_d = 0
        best_pair = (0, 0)
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > best_d:
                            best_d = dist[nb]
                            best_pair = (s, nb)
                        q.append(nb)
        return best_d, best_pair
    
    def greedy_fill(edge_set, deadline):
        """Greedy: repeatedly add the edge that most reduces the farthest pair distance."""
        es = set(edge_set)
        adj = build_adj(es)
        added = []
        
        while time.time() < deadline:
            diam, (s, t) = get_farthest_pair(adj)
            if diam <= 4:
                break
            
            # BFS from s and t
            def bfs_from(start):
                dist = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    node = q.popleft()
                    for nb in adj[node]:
                        if dist[nb] == -1:
                            dist[nb] = dist[node] + 1
                            q.append(nb)
                return dist
            
            dist_s = bfs_from(s)
            dist_t = bfs_from(t)
            
            best_cand = None
            best_new_dist = float('inf')
            
            for u, v in candidates:
                key = (min(u, v), max(u, v))
                if key in es:
                    continue
                # Distance s->t via (u,v): min(d(s,u)+1+d(v,t), d(s,v)+1+d(u,t))
                d1 = dist_s[u] + 1 + dist_t[v]
                d2 = dist_s[v] + 1 + dist_t[u]
                cand_dist = min(d1, d2)
                
                if cand_dist < best_new_dist:
                    best_new_dist = cand_dist
                    best_cand = key
            
            if best_cand is None:
                break
            
            u, v = best_cand
            es.add(best_cand)
            adj[u].add(v)
            adj[v].add(u)
            added.append([u, v])
        
        return es, added
    
    def remove_redundant(es, deadline):
        """Try to remove edges while keeping diameter <= 4."""
        es = set(es)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for e in list(es):
                es2 = es - {e}
                adj = build_adj(es2)
                if bfs_diameter(adj) <= 4:
                    es = es2
                    improved = True
                    break  # restart scan after removal
        return es
    
    deadline = time.time() + 70  # safety margin before 90s kill
    
    best_result = None
    best_count = float('inf')
    
    # Strategy 1: Pure greedy from base
    es, added = greedy_fill(set(base_edges), deadline)
    if bfs_diameter(build_adj(es)) <= 4:
        es = remove_redundant(es, deadline)
        if len(es) < best_count:
            best_count = len(es)
            best_result = es
    
    # Strategy 2: Hub-based - try several hub positions
    # A hub at position h connects to many vertices
    # For triangle-free, hub must connect to opposite parity only
    # Try hubs at various positions
    for hub in range(n):
        if time.time() > deadline:
            break
        
        # Build initial edge set: base + hub connections to strategic vertices
        # Connect hub to vertices that are far apart in the path
        # hub parity determines which vertices it can connect to
        hub_parity = hub % 2
        
        # Sort candidate neighbors of hub by distance from hub (descending)
        hub_candidates = []
        for v in range(n):
            if v == hub:
                continue
            if (v % 2) != hub_parity:  # opposite parity
                key = (min(hub, v), max(hub, v))
                if key not in base_edges:
                    hub_candidates.append((abs(v - hub), v, key))
        
        hub_candidates.sort(reverse=True)  # farthest first
        
        # Try adding top k hub edges + greedy fill
        for k in range(1, min(len(hub_candidates), 12) + 1):
            if time.time() > deadline:
                break
            
            initial = set(base_edges)
            for _, _, key in hub_candidates[:k]:
                initial.add(key)
            
            es, added = greedy_fill(initial, deadline)
            if bfs_diameter(build_adj(es)) <= 4:
                es = remove_redundant(es, deadline)
                if len(es) < best_count:
                    best_count = len(es)
                    best_result = es
    
    # Strategy 3: Two-hub - try pairs of hubs
    if time.time() < deadline:
        for h1 in range(4, n - 4, 2):  # even positions
            for h2 in range(h1 + 2, n - 4, 2):  # even positions, different
                if time.time() > deadline:
                    break
                
                initial = set(base_edges)
                
                # Add edges from h1 to far opposite-parity vertices
                for v in range(n):
                    if v == h1 or v == h2:
                        continue
                    if (v % 2) != (h1 % 2):
                        key = (min(h1, v), max(h1, v))
                        if key not in base_edges and abs(v - h1) >= 4:
                            initial.add(key)
                
                # Add edges from h2 to far opposite-parity vertices
                for v in range(n):
                    if v == h1 or v == h2:
                        continue
                    if (v % 2) != (h2 % 2):
                        key = (min(h2, v), max(h2, v))
                        if key not in base_edges and abs(v - h2) >= 4:
                            initial.add(key)
                
                # Ensure h1-h2 connection if possible
                if (h1 + h2) % 2 == 1:
                    key = (min(h1, h2), max(h1, h2))
                    if key not in base_edges:
                        initial.add(key)
                
                es, added = greedy_fill(initial, deadline)
                if bfs_diameter(build_adj(es)) <= 4:
                    es = remove_redundant(es, deadline)
                    if len(es) < best_count:
                        best_count = len(es)
                        best_result = es
    
    # Fallback: if no valid result found, do pure greedy
    if best_result is None:
        es, added = greedy_fill(set(base_edges), time.time() + 5)
        best_result = es
    
    # Convert to list of [u, v]
    result = []
    for u, v in best_result:
        result.append([u, v])
    
    return result
