import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        if u > v: u, v = v, u
        base_edges.add((u, v))
    
    # Generate valid candidates: opposite parity, not in base
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u + v) % 2 == 1 and (u, v) not in base_edges:
                candidates.append((u, v))
                
    def compute_diameter(edges_set):
        adj = {i: [] for i in range(n)}
        for u, v in edges_set:
            adj[u].append(v)
            adj[v].append(u)
            
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > max_dist:
                            max_dist = dist[nb]
                        q.append(nb)
        return max_dist

    current_edges = set(base_edges)
    added = []
    
    # Greedy loop with exact diameter evaluation
    # Safety limit
    max_iterations = 50 
    for _ in range(max_iterations):
        diam = compute_diameter(current_edges)
        if diam <= 4:
            break
        
        best_cand = None
        best_new_diam = float('inf')
        # Tie breaker: maybe min sum of distances? Or just first.
        # Let's add a secondary metric: number of pairs at max distance?
        # Or just keep it simple.
        
        for u, v in candidates:
            if (u, v) in current_edges:
                continue
            
            # Tentatively add
            new_edges = current_edges | {(u, v)}
            new_diam = compute_diameter(new_edges)
            
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_cand = (u, v)
                
        if best_cand is None:
            # No candidate improves diameter?
            # This shouldn't happen if graph is connected and diam > 4
            # Unless we are stuck.
            break
            
        current_edges.add(best_cand)
        added.append(list(best_cand))
        
    return added
