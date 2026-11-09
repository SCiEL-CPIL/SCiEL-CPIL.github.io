"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
        base_edges.add((max(e[0], e[1]), min(e[0], e[1])))
    
    def check_valid(added_edges):
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        added_set = set()
        for u, v in added_edges:
            if u == v:
                return False
            if (min(u, v), max(u, v)) in base_edges:
                return False
            if (min(u, v), max(u, v)) in added_set:
                return False
            added_set.add((min(u, v), max(u, v)))
            adj[u].add(v)
            adj[v].add(u)
        
        # Check triangle-free
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    common = adj[u] & adj[v]
                    if common:
                        return False
        
        # Check connected via BFS from 0
        visited = set([0])
        queue = [0]
        while queue:
            cur = queue.pop(0)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(visited) != n:
            return False
        
        # Check diameter <= 4
        for start in range(n):
            dist = {start: 0}
            queue = [start]
            while queue:
                cur = queue.pop(0)
                for nb in adj[cur]:
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        if dist[nb] <= 4:
                            queue.append(nb)
            if len(dist) != n:
                return False
            for d in dist.values():
                if d > 4:
                    return False
        
        return True
    
    # Candidate construction: add edges that create a "grid-like" structure
    # Strategy: Add edges of length 4 and length 8 (or similar) to create shortcuts
    # While maintaining triangle-freeness
    
    # Let's try a systematic approach:
    # The path has diameter 23. We need diameter 4.
    # With 24 nodes, we need edges that create many shortcuts.
    
    # Approach: Add all edges (i, i+4) and (i, i+8) that don't create triangles
    # Then verify and potentially add more
    
    candidate1 = []
    for i in range(n):
        if i + 4 < n:
            candidate1.append((i, i + 4))
    
    if not check_valid(candidate1):
        # Try different lengths
        candidate1 = []
        for i in range(n):
            if i + 5 < n:
                candidate1.append((i, i + 5))
    
    if not check_valid(candidate1):
        candidate1 = []
        for i in range(n):
            if i + 6 < n:
                candidate1.append((i, i + 6))
    
    if not check_valid(candidate1):
        # Try mixed approach
        candidate1 = []
        for i in range(n):
            if i + 4 < n and i % 2 == 0:
                candidate1.append((i, i + 4))
            if i + 6 < n and i % 2 == 1:
                candidate1.append((i, i + 6))
    
    # Verify the best candidate we have
    if check_valid(candidate1):
        return [[int(u), int(v)] for u, v in candidate1]
    
    # Fallback: use the original construction if needed
    return [[u, v] for u in range(n) for v in range(u + 1, n) if (v - u) % 2 == 1 and (min(u, v), max(u, v)) not in base_edges]
# EVOLVE-BLOCK-END
