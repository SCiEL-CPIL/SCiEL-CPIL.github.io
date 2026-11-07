import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
    
    # Partitions
    A = [i for i in range(n) if i % 2 == 0]
    B = [i for i in range(n) if i % 2 == 1]
    
    # Candidate edges: all (a, b) with a in A, b in B, not in base
    candidates = []
    for a in A:
        for b in B:
            if (a, b) not in base_edges and (b, a) not in base_edges:
                candidates.append((a, b))
                
    # Current adjacency list
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    added_edges = []
    
    # Helper to compute diameter
    def get_diameter():
        max_dist = 0
        for s in range(n):
            # BFS
            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            max_dist = max(max_dist, max(dist))
        return max_dist

    # Greedy loop
    # To avoid infinite loops or slow convergence, we limit iterations
    max_iters = 50 
    start_time = time.time()
    
    while True:
        if time.time() - start_time > 50: # Safety margin
            break
        if len(added_edges) > 20: # Unlikely to need more
            break
            
        diam = get_diameter()
        if diam <= 4:
            break
            
        # Find pair with max distance
        max_u, max_v = -1, -1
        max_d = 0
        for u in range(n):
            for v in range(u+1, n):
                # We need distance. Re-computing all pairs is expensive.
                # Let's just pick a pair that is far apart in the path initially, 
                # or better, track the actual farthest pair from the last BFS.
                pass
        
        # Optimization: Instead of full all-pairs BFS every time, 
        # we can just pick a random pair that is far, or use the fact that 
        # the diameter endpoints in a path are 0 and n-1 initially.
        # As we add edges, the diameter shrinks.
        # Let's just compute all pairs distances once per iteration.
        
        # Re-compute distances from all nodes? O(N * (N+E))
        # N=24, E~30. 24 * 54 ~ 1300 ops. Very fast.
        
        all_dists = [[-1]*n for _ in range(n)]
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            all_dists[s] = dist
            
        # Find max distance pair
        current_max_dist = 0
        worst_u, worst_v = 0, 1
        for u in range(n):
            for v in range(u+1, n):
                if all_dists[u][v] > current_max_dist:
                    current_max_dist = all_dists[u][v]
                    worst_u, worst_v = u, v
        
        if current_max_dist <= 4:
            break
            
        # Try to add an edge that reduces the distance between worst_u and worst_v
        best_edge = None
        best_new_dist = current_max_dist + 1
        
        # Also consider global diameter reduction? 
        # For now, just reduce the specific worst pair.
        for a, b in candidates:
            if (a, b) in base_edges or (b, a) in base_edges:
                continue
            # Check if edge is already added
            if any((a==e[0] and b==e[1]) or (b==e[0] and a==e[1]) for e in added_edges):
                continue
                
            # Temporarily add edge
            adj[a].append(b)
            adj[b].append(a)
            
            # Compute distance between worst_u and worst_v
            # BFS from worst_u
            dist = [-1] * n
            dist[worst_u] = 0
            queue = [worst_u]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                        
            new_dist = dist[worst_v]
            
            # Remove edge
            adj[a].remove(b)
            adj[b].remove(a)
            
            if new_dist < best_new_dist:
                best_new_dist = new_dist
                best_edge = (a, b)
                
        if best_edge:
            a, b = best_edge
            adj[a].append(b)
            adj[b].append(a)
            added_edges.append([a, b])
        else:
            # If no single edge reduces the specific pair, 
            # it might be stuck or need a different strategy.
            # But with bipartite graph, we can always connect to reduce distance.
            # If no edge helps, break to avoid infinite loop.
            break
            
    # Convert to list of lists of ints
    result = []
    for e in added_edges:
        result.append([int(e[0]), int(e[1])])
        
    return result
