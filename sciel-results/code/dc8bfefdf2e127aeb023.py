import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    n = 16
    # Construct K(8,8): complete bipartite graph on parts {0..7} and {8..15}
    # This is triangle-free with diameter 2, so every edge is critical.
    # By Turan's theorem, 64 is the maximum edges for a triangle-free graph on 16 vertices.
    edges = []
    for a in range(8):
        for b in range(8, 16):
            edges.append([a, b])

    # Verify: triangle-free + diameter 2 implies all edges critical
    if _verify(edges, n):
        return edges

    # Fallback: Star graph (valid but lower edge count)
    return [[0, v] for v in range(1, n)]

def _verify(edges, n):
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Check diameter is exactly 2 using BFS
    diam = _get_diameter(adj, n)
    if diam != 2:
        return False

    # For triangle-free diameter-2 graphs, every edge is critical:
    # removing edge (u,v) means u,v have no common neighbor (triangle-free),
    # so dist(u,v) >= 3 in the modified graph, making diameter >= 3.
    # Verify triangle-freeness as a sanity check.
    for u in range(n):
        for v in adj[u]:
            if v > u:  # avoid double check
                # Check if u and v share any common neighbor
                common = adj[u] & adj[v]
                if common:
                    return False

    return True

def _get_diameter(adj, n):
    max_d = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = [s]
        i = 0
        while i < len(q):
            u = q[i]
            i += 1
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    q.append(w)
        if -1 in dist:
            return None  # Disconnected
        max_d = max(max_d, max(dist))
    return max_d
