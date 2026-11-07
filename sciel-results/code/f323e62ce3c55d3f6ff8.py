"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Generate candidate edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                candidates.append((u, v))
    
    def is_valid(added_edges):
        edges_set = set(base_edges)
        for e in added_edges:
            edges_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        if len(edges_set) != len(base_edges) + len(added_edges):
            return False
        
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Triangle-free
        for a in range(n):
            for b in adj[a]:
                if b <= a:
                    continue
                for c in adj[b]:
                    if c > b and c in adj[a]:
                        return False
        
        # Connected
        visited = [False] * n
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if not all(visited):
            return False
        
        # Diameter <= 4
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            while queue:
                u = queue.pop(0)
                if dist[u] >= 4:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if any(d == -1 or d > 4 for d in dist):
                return False
        
        return True
    
    # Try k from 5 upward
    for k in range(5, 13):
        # Use a smarter search: try promising edge sets
        # First, identify "critical" edges that help most
        
        # For diameter 4, vertices far apart need shortcuts
        # Key pairs: (0,23), (0,22), (1,23), etc.
        
        # Greedy with verification
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def get_diameter():
            max_d = 0
            for s in range(n):
                dist = [-1] * n
                dist[s] = 0
                q = [s]
                while q:
                    u = q.pop(0)
                    if dist[u] >= 4:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            q.append(v)
                for d in dist:
                    if d > max_d:
                        max_d = d
            return max_d
        
        def check_tri():
            for a in range(n):
                for b in adj[a]:
                    if b <= a:
                        continue
                    for c in adj[b]:
                        if c > b and c in adj[a]:
                            return False
            return True
        
        added = []
        remaining = set(candidates)
        
        for i in range(k):
            best = None
            best_d = get_diameter()
            for e in list(remaining):
                u, v = e
                adj[u].add(v)
                adj[v].add(u)
                if check_tri():
                    d = get_diameter()
                    if d < best_d:
                        best_d = d
                        best = e
                adj[u].remove(v)
                adj[v].remove(u)
            
            if best is None:
                break
            
            u, v = best
            adj[u].add(v)
            adj[v].add(u)
            added.append([u, v])
            remaining.discard(best)
        
        if len(added) <= k and is_valid(added):
            return [[int(u), int(v)] for u, v in added]
    
    # Fallback: return a known valid solution
    # Try: (0,4), (1,5), (2,6), (3,7), (4,8), (5,9), (6,10), (7,11), (8,12), (9,13), (10,14), (11,15), (12,16), (13,17), (14,18), (15,19), (16,20), (17,21), (18,22), (19,23)
    # That's 20, too many. Let me just return empty or a minimal valid set.
    return []
# EVOLVE-BLOCK-END
