import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    
    # Build base adjacency (path)
    adj = [set() for _ in range(n)]
    for i in range(n - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    
    # Base edges set for checking
    base_edges = set()
    for i in range(n - 1):
        base_edges.add((i, i + 1))
        base_edges.add((i + 1, i))
    
    def compute_diameter():
        """Compute diameter via BFS from each vertex."""
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            m = max(dist)
            if m > max_dist:
                max_dist = m
        return max_dist
    
    def compute_max_dist_exceeding_4():
        """Return the maximum distance if it exceeds 4, else -1. Faster: early exit."""
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return dist[v]
                        q.append(v)
        return -1
    
    # Candidate edges: only cross-partition (even-odd) edges that aren't base
    # This guarantees triangle-freeness since path is bipartite
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges and (v, u) not in base_edges:
                # Cross-partition: one even, one odd
                if (u % 2) != (v % 2):
                    candidates.append((u, v))
    
    added = []
    
    # Phase 1: Greedy - add edges that most reduce the "worst" distance
    current_diam = compute_diameter()
    
    while current_diam > 4 and time.time() < deadline:
        best_edge = None
        best_worst = current_diam + 1
        best_sum = float('inf')
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            
            adj[u].add(v)
            adj[v].add(u)
            
            # Compute worst distance and sum
            worst = -1
            total = 0
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                q = deque([src])
                while q:
                    x = q.popleft()
                    for y in adj[x]:
                        if dist[y] == -1:
                            dist[y] = dist[x] + 1
                            q.append(y)
                for d in dist:
                    if d != -1:
                        total += d
                        if d > worst:
                            worst = d
            
            adj[u].remove(v)
            adj[v].remove(u)
            
            if worst < best_worst or (worst == best_worst and total < best_sum):
                best_worst = worst
                best_sum = total
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
        candidates.remove(best_edge)
        current_diam = best_worst
    
    # Phase 2: Try to remove edges
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(added)):
            if time.time() > deadline:
                break
            u1, v1 = added[i]
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            d = compute_diameter()
            if d <= 4:
                added.pop(i)
                candidates.append((u1, v1))
                improved = True
                break
            else:
                adj[u1].add(v1)
                adj[v1].add(u1)
    
    # Phase 3: Try swaps (remove 2, add 1)
    if time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            found_swap = False
            for i in range(len(added)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(added)):
                    if time.time() > deadline:
                        break
                    u1, v1 = added[i]
                    u2, v2 = added[j]
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    
                    best_new = None
                    for u3, v3 in candidates:
                        if time.time() > deadline:
                            break
                        adj[u3].add(v3)
                        adj[v3].add(u3)
                        d = compute_diameter()
                        if d <= 4:
                            best_new = (u3, v3)
                            break
                        adj[u3].remove(v3)
                        adj[v3].remove(u3)
                    
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    
                    if best_new is not None:
                        u3, v3 = best_new
                        adj[u1].remove(v1)
                        adj[v1].remove(u1)
                        adj[u2].remove(v2)
                        adj[v2].remove(u2)
                        added.pop(j)
                        added.pop(i)
                        adj[u3].add(v3)
                        adj[v3].add(u3)
                        added.append([u3, v3])
                        candidates.remove(best_new)
                        candidates.append((u1, v1))
                        candidates.append((u2, v2))
                        found_swap = True
                        break
                if found_swap:
                    break
            if not found_swap:
                break
    
    # Final validation
    result = []
    for edge in added:
        result.append([int(edge[0]), int(edge[1])])
    return result
