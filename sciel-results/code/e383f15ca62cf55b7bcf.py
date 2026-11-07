"""Rewritten program for triangle-free diameter-four augmentation.
Strategy: Exhaustive search over carefully pruned candidate pool.
The base graph is a path on 24 vertices. To achieve diameter <= 4 while
maintaining triangle-free property, we can only add edges between vertices
of opposite parity.
"""
# EVOLVE-BLOCK-START
import time
from collections import deque
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 90s limit
    
    n = P['n']
    base_edges_set = set()
    for u, v in P['edges']:
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
    
    # Build initial adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Candidate edges: opposite parity (to maintain bipartiteness/triangle-free), not in base
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 == 1 and (u, v) not in base_edges_set:
                candidates.append((u, v))
    
    def get_all_distances(adj):
        """BFS from each vertex to compute all-pairs distances."""
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
    
    def get_diameter(dist):
        """Maximum distance in the distance matrix."""
        max_d = 0
        for i in range(n):
            for j in range(n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
        return max_d
    
    # Sort candidates by span descending - long edges reduce diameter most effectively
    candidates_sorted = sorted(candidates, key=lambda x: abs(x[0] - x[1]), reverse=True)
    
    # We need a pool of edges that are diverse and high-impact.
    # For n=24, D=4, we expect the answer to be around 3-4 edges.
    # Let's try a pool of top 15-20 edges and search combinations.
    
    # Precompute which edges are "useful" - those that connect far apart vertices
    # and are not too local.
    
    # Strategy: Try k=3 first, then k=4, then k=5
    # Use a pool of the top ~15 edges by span
    
    best_solution = None
    
    # Pool selection: take edges with the largest spans, ensuring diversity
    # Top spans for n=24: (0,23), (1,22), (0,21), (2,23), (1,21), (3,22), etc.
    
    # Let's use a pool of top 12 edges
    pool = candidates_sorted[:12]
    
    # Try k = 3
    for k in range(3, 6):
        if time.time() > deadline:
            break
            
        # For k=3: C(12,3) = 220
        # For k=4: C(12,4) = 495
        # For k=5: C(12,5) = 792
        # Total: ~1500 BFS checks, each O(n*(n+e)) which is fast for n=24
        
        for combo in itertools.combinations(pool, k):
            if time.time() > deadline:
                break
                
            # Build test adjacency
            test_adj = [set(s) for s in adj]
            for u, v in combo:
                test_adj[u].add(v)
                test_adj[v].add(u)
                
            # Check triangle-free: verify no triangle exists
            # Since we only add edges between opposite parity, the graph remains bipartite,
            # hence triangle-free. But let's be safe and skip explicit check since
            # opposite parity guarantees bipartiteness.
            
            # Check diameter
            dist = get_all_distances(test_adj)
            diam = get_diameter(dist)
            
            if diam <= 4:
                best_solution = list(combo)
                break
        
        if best_solution is not None:
            break
    
    # If exhaustive search over the top pool didn't work, expand the pool
    if best_solution is None:
        if time.time() < deadline:
            pool = candidates_sorted[:20]
            for k in range(3, 6):
                if time.time() > deadline:
                    break
                for combo in itertools.combinations(pool, k):
                    if time.time() > deadline:
                        break
                    test_adj = [set(s) for s in adj]
                    for u, v in combo:
                        test_adj[u].add(v)
                        test_adj[v].add(u)
                    dist = get_all_distances(test_adj)
                    diam = get_diameter(dist)
                    if diam <= 4:
                        best_solution = list(combo)
                        break
                if best_solution is not None:
                    break
    
    # Fallback: Greedy approach if exhaustive search fails
    if best_solution is None:
        current_edges = set()
        current_adj = [set(s) for s in adj]
        current_diam = 23
        
        while current_diam > 4:
            if time.time() > deadline:
                break
            best_edge = None
            best_diam = current_diam + 1
            
            for u, v in candidates:
                if (u, v) in current_edges or (v, u) in current_edges:
                    continue
                # Check if adding this edge creates a triangle
                # Since we only use opposite parity edges, triangles are impossible
                current_adj[u].add(v)
                current_adj[v].add(u)
                new_diam = get_diameter(get_all_distances(current_adj))
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_edge = (u, v)
                if new_diam <= 4:
                    break
            
            if best_edge is None:
                break
                
            u, v = best_edge
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_edges.add((u, v))
            current_edges.add((v, u))
            current_diam = best_diam
        
        added = []
        for u, v in current_edges:
            if (u, v) not in base_edges_set and (v, u) not in base_edges_set:
                if u < v:
                    added.append([u, v])
                elif v < u:
                    added.append([v, u])
        
        final_added = []
        seen = set()
        for e in added:
            key = (e[0], e[1])
            if key not in seen:
                final_added.append([int(e[0]), int(e[1])])
                seen.add(key)
        return final_added
    
    final_added = [[int(u), int(v)] for u, v in best_solution]
    return final_added

# EVOLVE-BLOCK-END
