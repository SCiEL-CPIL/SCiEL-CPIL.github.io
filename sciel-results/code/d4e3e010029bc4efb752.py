"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _check_validity(added_edges):
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    all_edges = set(base_edges)
    for e in added_edges:
        u, v = int(min(e[0], e[1])), int(max(e[0], e[1]))
        if (u, v) in all_edges:
            return False
        all_edges.add((u, v))
    
    adj = [[] for _ in range(n)]
    for u, v in all_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Triangle-free check
    for u in range(n):
        neighbors = adj[u]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in adj[neighbors[i]]:
                    return False
    
    # Connected check
    visited = [False] * n
    visited[0] = True
    queue = [0]
    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)
    if not all(visited):
        return False
    
    # Diameter <= 4 check
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        while queue:
            node = queue.pop(0)
            if dist[node] >= 4:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        if any(d == -1 for d in dist):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80

    fallback = [
        [12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
        [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]
    ]

    n = 24
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))

    # Local search: try to find 9-edge solution by swapping one edge
    current = [list(e) for e in fallback]

    all_non_base = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                all_non_base.append((u, v))

    for i in range(len(current)):
        if time.time() > deadline:
            break
        removed = current[i]
        remaining = [e for j, e in enumerate(current) if j != i]
        remaining_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in remaining)
        for cand in all_non_base:
            if cand in remaining_set:
                continue
            if cand == (min(removed[0], removed[1]), max(removed[0], removed[1])):
                continue
            test = remaining + [[int(cand[0]), int(cand[1])]]
            if _check_validity(test):
                return test

    return fallback
# EVOLVE-BLOCK-END
