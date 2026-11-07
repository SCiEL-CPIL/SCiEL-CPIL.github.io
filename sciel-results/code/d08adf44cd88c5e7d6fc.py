"""Rewritten program: dense bipartite construction avoiding equal-degree 3-path endpoints."""
import time

# EVOLVE-BLOCK-START
P = {'n': 15}

def construct():
    start = time.time()
    deadline = start + 75  # safety margin before 90s kill

    n = P['n']
    # Split into two parts of sizes 7 and 8
    # Part A: vertices 0..6 (size 7, degree 8)
    # Part B: vertices 7..14 (size 8, degree 7)
    size_a = 7
    size_b = 8
    assert size_a + size_b == n

    edges = []
    for i in range(size_a):
        for j in range(size_a, n):
            edges.append([i, j])

    # Validate the construction
    # Build adjacency and degrees
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]

    # Check: for every edge (u,v), and every s in adj[u] (s != v), and every t in adj[v] (t != u),
    # if s != t, then deg[s] != deg[t]
    for u, v in edges:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u:
                    continue
                if s == t:
                    continue
                if deg[s] == deg[t]:
                    # This shouldn't happen with K_{7,8}, but guard just in case
                    raise ValueError(f"Invalid: path {s}-{u}-{v}-{t} has equal degrees {deg[s]}")

    # Ensure types are exact Python ints and lists
    result = []
    for u, v in edges:
        result.append([int(u), int(v)])

    return result
# EVOLVE-BLOCK-END
