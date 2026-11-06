import time
from collections import defaultdict, deque

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def min_t_join(edges, n):
        deg = defaultdict(int)
        adj = defaultdict(set)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        T = [v for v in range(n) if deg[v] % 2 == 1]
        if not T:
            return set()
        
        # BFS from each node in T to get distances to other nodes in T
        # Since N is small, just BFS from all nodes in T
        dist = {t: {} for t in T}
        prev = {t: {} for t in T}
        
        for start in T:
            dist[start][start] = 0
            prev[start][start] = None
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in dist[start]:
                        dist[start][v] = dist[start][u] + 1
                        prev[start][v] = u
                        q.append(v)
                        
        # Greedy Matching
        m = len(T)
        used = [False] * m
        pairs = []
        for i in range(m):
            for j in range(i+1, m):
                u, v = T[i], T[j]
                if v in dist[u]:
                    pairs.append((dist[u][v], i, j))
        
        pairs.sort()
        
        F = set()
        for d, i, j in pairs:
            if not used[i] and not used[j]:
                used[i] = True
                used[j] = True
                u, v = T[i], T[j]
                # Reconstruct path
                curr = v
                while curr != u:
                    p = prev[u][curr]
                    F.add(tuple(sorted((curr, p))))
                    curr = p
        return F

    def decompose_graph(edges, n):
        F = min_t_join(edges, n)
        H = set(tuple(sorted(e)) for e in edges) - F
        
        cycles = []
        remaining_H = set(H)
        
        while remaining_H:
            # Build adj for remaining_H
            adj_H = defaultdict(set)
            for a, b in remaining_H:
                adj_H[a].add(b)
                adj_H[b].add(a)
                
            # Pick an edge
            u, v = next(iter(remaining_H))
            
            # Find path from v to u in remaining_H
            # BFS
            visited = {v}
            parent = {v: None}
            q = deque([v])
            found = False
            while q:
                curr = q.popleft()
                if curr == u:
                    found = True
                    break
                for neighbor in adj_H[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = curr
                        q.append(neighbor)
                        
            if found:
                path = [u]
                curr = v
                while curr != u:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                cycles.append(path)
                
                for i in range(len(path) - 1):
                    a, b = path[i], path[i+1]
                    remaining_H.remove(tuple(sorted((a, b))))
            else:
                break
                
        pieces = cycles
        for e in F:
            pieces.append(list(e))
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - t0 > DEADLINE - 20:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        n = spec[0]
        pieces = decompose_graph(edges, n)
        results.append(pieces)
    return results
