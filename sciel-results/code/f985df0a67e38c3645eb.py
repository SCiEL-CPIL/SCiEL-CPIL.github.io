import time

P = {'n': 15}

def construct():
    n = P['n']
    
    def is_valid(edges):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    best_edges = []
    for cutoff in range(0, 2 * n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if i + j >= cutoff:
                    edges.append([i, j])
        if is_valid(edges) and len(edges) > len(best_edges):
            best_edges = edges
    
    return [[int(u), int(v)] for u, v in best_edges]
