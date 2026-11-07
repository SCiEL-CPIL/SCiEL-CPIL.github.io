import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = 24
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add(frozenset((u, v)))
    
    # Bipartition: even in A, odd in B
    A = set(range(0, n, 2))
    B = set(range(1, n, 2))
    
    # Candidate edges: between A and B, not already in base
    candidates = []
    for u in A:
        for v in B:
            if frozenset((u, v)) not in base_edges:
                candidates.append((u, v))
    
    def _check_valid(added_edges, n, base_edges):
        adj = [set() for _ in range(n)]
        for e in base_edges:
            u, v = tuple(e)
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Triangle-free check
        for i in range(n):
            nbrs = list(adj[i])
            for a in range(len(nbrs)):
                for b in range(a + 1, len(nbrs)):
                    if nbrs[b] in adj[nbrs[a]]:
                        return False
        
        # Diameter check
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= 4:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d == -1 or d > 4:
                    return False
        
        return True
    
    # Fallback: 7-edge hub solution
    hub = 11
    targets = [1, 4, 7, 15, 18, 21, 23]
    fallback = [[hub, t] for t in targets]
    
    # Verify fallback is valid
    if not _check_valid(fallback, n, base_edges):
        # This shouldn't happen, but return empty if invalid
        return []
    
    # Try to find solutions with fewer than 7 edges
    deadline = time.time() + 85  # 85s deadline (5s safety margin)
    
    for k in range(1, 7):
        if time.time() > deadline:
            break
        
        found = None
        
        if k <= 4 and len(candidates) < 5000:
            # For small k, try exhaustive search
            for combo in combinations(candidates, k):
                if time.time() > deadline:
                    break
                if _check_valid(combo, n, base_edges):
                    found = list(combo)
                    break
        else:
            # Random sampling for larger k
            import random
            rng = random.Random(42 + k)
            max_iters = 200000
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                combo = rng.sample(candidates, k)
                if _check_valid(combo, n, base_edges):
                    found = list(combo)
                    break
        
        if found:
            return [[u, v] for u, v in found]
    
    # Return the 7-edge fallback
    return fallback

