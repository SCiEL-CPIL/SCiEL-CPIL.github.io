import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Two disjoint K_{3,3} components
    # Cycle lengths in K_{3,3}: {4, 6}
    # Total distinct cycle lengths: 2

    edges = []

    # Component 1: K_{3,3} on vertices 0..5
    # Part A = {0, 1, 2}, Part B = {3, 4, 5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])

    # Component 2: K_{3,3} on vertices 6..11
    # Part A = {6, 7, 8}, Part B = {9, 10, 11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])

    # Verification: degrees
    n = 12
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d < 3 for d in deg):
        # Should not happen, but return empty as invalid signal
        return []

    # Verification: triangle-free (bipartite check via 2-coloring each component)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Check for triangles: for each edge (u,v), check if they share a common neighbor
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if adj[u] & adj[v]:
                    # Triangle found - should not happen for bipartite K_{3,3}
                    return []

    # Verify it's bipartite (confirms no odd cycles, hence no triangles)
    color = [-1] * n
    for start in range(n):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = [start]
        while queue:
            node = queue.pop()
            for nb in adj[node]:
                if color[nb] == -1:
                    color[nb] = color[node] ^ 1
                    queue.append(nb)
                elif color[nb] == color[node]:
                    # Odd cycle found - invalid
                    return []

    # Verify contains a cycle: K_{3,3} has 4-cycles, guaranteed
    # (edges count = 18, vertices = 12, so cycles exist)

    # Ensure proper types: Python int, list of lists
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])

    if time.time() > deadline:
        return result

    return result
