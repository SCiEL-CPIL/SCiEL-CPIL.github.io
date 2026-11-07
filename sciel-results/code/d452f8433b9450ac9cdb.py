import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    existing = set()
    for u, v in base_edges:
        existing.add((min(u, v), max(u, v)))
    
    def validate(added):
        # Check no existing edges, no duplicates
        edge_set = set()
        for u, v in added:
            mn, mx = min(u, v), max(u, v)
            if (mn, mx) in existing:
                return False
            if (mn, mx) in edge_set:
                return False
            edge_set.add((mn, mx))
        
        # Build full adjacency
        adj2 = [set(adj[i]) for i in range(n)]
        for u, v in added:
            adj2[u].add(v)
            adj2[v].add(u)
        
        # Triangle-free check
        for u in range(n):
            for v in adj2[u]:
                if v > u:
                    if adj2[u] & adj2[v]:
                        return False
        
        # Connected check
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in adj2[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if not all(visited):
            return False
        
        # Diameter <= 4
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in adj2[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if any(d > 4 for d in dist):
                return False
        
        return True
    
    # Try the 6-edge solution
    six_edge = [(6, 0), (6, 12), (6, 17), (17, 6), (17, 23), (17, 11)]
    # Remove duplicate (6,17) and (17,6)
    six_edge = [(6, 0), (6, 12), (6, 17), (17, 23), (17, 11), (17, 10)]
    
    if validate(six_edge):
        return [[int(u), int(v)] for u, v in six_edge]
    
    # Try other 6-edge configurations
    # Two hubs: 6 and 17
    candidates_6 = [
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 11), (17, 10)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 11), (17, 12)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 10), (17, 13)],
        [(5, 0), (5, 11), (5, 16), (16, 23), (16, 10), (16, 12)],
        [(6, 0), (6, 11), (6, 17), (17, 23), (17, 12), (17, 10)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 11), (17, 14)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 9), (17, 12)],
        [(7, 0), (7, 13), (7, 17), (17, 23), (17, 11), (17, 10)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 11), (17, 15)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 10), (17, 11)],
    ]
    
    for sol in candidates_6:
        if validate(sol):
            return [[int(u), int(v)] for u, v in sol]
    
    # Try 7-edge solutions
    seven_edge = [(6, 0), (6, 12), (6, 17), (17, 23), (17, 11), (17, 10), (17, 13)]
    if validate(seven_edge):
        return [[int(u), int(v)] for u, v in seven_edge]
    
    # Fallback: known working solution
    fallback = [[11, 1], [11, 4], [11, 7], [11, 15], [11, 18], [11, 21], [11, 23]]
    return [[int(u), int(v)] for u, v in fallback]
