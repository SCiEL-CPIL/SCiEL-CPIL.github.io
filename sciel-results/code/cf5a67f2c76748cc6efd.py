"""Triangle-free diameter-four augmentation via set-cover-inspired search."""
import time
from collections import deque
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = P['n']
    base_edges_set = set()
    for u, v in P['edges']:
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
    
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # All candidate edges: opposite parity, not in base
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 == 1 and (u, v) not in base_edges_set:
                candidates.append((u, v))
    
    def compute_diameter(adj_list):
        """Compute diameter using BFS from each node."""
        max_diam = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj_list[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for d in dist:
                if d > max_diam:
                    max_diam = d
        return max_diam
    
    def get_pairs_over_limit(adj_list, limit=4):
        """Get all pairs with distance > limit."""
        over = set()
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj_list[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for j in range(src + 1, n):
                if dist[j] > limit:
                    over.add((src, j))
        return over
    
    # Score candidates by how many "critical pairs" they help
    # A pair (i,j) with dist > 4 is helped by edge (u,v) if adding it reduces dist(i,j)
    def edge_helps_pair(adj_list, edge, pair):
        u, v = edge
        i, j = pair
        # Temporarily add edge
        adj_list[u].add(v)
        adj_list[v].add(u)
        
        # BFS from i to j
        dist = [-1] * n
        dist[i] = 0
        q = deque([i])
        while q:
            node = q.popleft()
            if node == j:
                break
            for nb in adj_list[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    q.append(nb)
        
        result = dist[j] <= 4
        
        # Remove edge
        adj_list[u].remove(v)
        adj_list[v].remove(u)
        return result
    
    # For efficiency, precompute which pairs are critical in base graph
    # and score each candidate edge by how many critical pairs it fixes when added alone
    base_crit_pairs = get_pairs_over_limit(adj, 4)
    
    edge_scores = []
    for e in candidates:
        test_adj = [set(s) for s in adj]
        u, v = e
        test_adj[u].add(v)
        test_adj[v].add(u)
        new_crit = get_pairs_over_limit(test_adj, 4)
        # Number of pairs that were critical before but are not critical after
        fixed = len(base_crit_pairs - new_crit)
        edge_scores.append((fixed, e))
    
    edge_scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Build a pool of top-scoring edges
    # Take top 20 by score, ensuring diversity
    pool = [e for _, e in edge_scores[:20]]
    
    # If pool is too small, add more
    if len(pool) < 12:
        for _, e in edge_scores[20:]:
            if e not in pool:
                pool.append(e)
            if len(pool) >= 12:
                break
    
    # Search for minimum k
    best_solution = None
    
    for k in range(3, 7):
        if time.time() > deadline:
            break
        
        # Use combinations from the pool
        # For k=3: C(20,3)=1140, k=4: C(20,4)=4845, k=5: C(20,5)=15504
        # This is manageable with efficient checking
        
        for combo in itertools.combinations(pool, k):
            if time.time() > deadline:
                break
            
            test_adj = [set(s) for s in adj]
            for u, v in combo:
                test_adj[u].add(v)
                test_adj[v].add(u)
            
            diam = compute_diameter(test_adj)
            if diam <= 4:
                best_solution = list(combo)
                break
        
        if best_solution is not None:
            break
    
    # Fallback: try a wider pool with k=3,4
    if best_solution is None and time.time() < deadline:
        pool2 = [e for _, e in edge_scores[:30]]
        for k in range(3, 5):
            if time.time() > deadline:
                break
            for combo in itertools.combinations(pool2, k):
                if time.time() > deadline:
                    break
                test_adj = [set(s) for s in adj]
                for u, v in combo:
                    test_adj[u].add(v)
                    test_adj[v].add(u)
                diam = compute_diameter(test_adj)
                if diam <= 4:
                    best_solution = list(combo)
                    break
            if best_solution is not None:
                break
    
    # Greedy fallback
    if best_solution is None:
        current_adj = [set(s) for s in adj]
        current_edges = set()
        
        while True:
            if time.time() > deadline:
                break
            diam = compute_diameter(current_adj)
            if diam <= 4:
                break
            
            best_edge = None
            best_diam = diam + 1
            
            for e in candidates:
                if e in current_edges:
                    continue
                u, v = e
                current_adj[u].add(v)
                current_adj[v].add(u)
                new_diam = compute_diameter(current_adj)
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_edge = e
            
            if best_edge is None:
                break
            
            u, v = best_edge
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_edges.add(e)
        
        best_solution = list(current_edges)
    
    if best_solution is None:
        return []
    
    # Format output
    final = []
    seen = set()
    for u, v in best_solution:
        key = (min(u, v), max(u, v))
        if key not in seen:
            final.append([int(u), int(v)])
            seen.add(key)
    
    return final
