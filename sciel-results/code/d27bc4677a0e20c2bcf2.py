import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))
    
    def compute_diameter(adj):
        max_diam = 0
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
            for d in dist:
                if d > max_diam:
                    max_diam = d
        return max_diam

    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_valid_solution(added_edges):
        adj = build_adj(added_edges)
        # Check triangle-free
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    if adj[u] & adj[v]:
                        return False
        diam = compute_diameter(adj)
        if diam > 4:
            return False
        return True

    # Candidate edges: only between even and odd (bipartite) to ensure triangle-free
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u % 2 == 0 and v % 2 == 1) or (u % 2 == 1 and v % 2 == 0):
                if (u, v) not in base_edges:
                    candidates.append((u, v))

    # Heuristic: Start with a hub-based solution
    # Try each vertex as a hub
    best_solution = None
    best_count = n * n

    for hub in range(n):
        # Connect hub to all opposite parity vertices at distance >= 3
        added = []
        for v in range(n):
            if v == hub: continue
            if (hub % 2) == (v % 2): continue
            if abs(hub - v) < 3: continue
            if (min(hub, v), max(hub, v)) not in base_edges:
                added.append((hub, v))
        
        added_list = [list(e) for e in added]
        if is_valid_solution(added_list):
            # Greedy removal
            current = added_list[:]
            removed = True
            while removed:
                removed = False
                for i in range(len(current)):
                    trial = current[:i] + current[i+1:]
                    if is_valid_solution(trial):
                        current = trial
                        removed = True
                        break
            if len(current) < best_count:
                best_count = len(current)
                best_solution = current[:]

    # If no hub solution found or we want to try mixed, we could add more search
    # But hub solution should find 10.
    
    return [[int(u), int(v)] for u, v in best_solution]
