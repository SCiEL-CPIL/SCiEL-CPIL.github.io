import time

def _bfs_maxdist(n, adj, exclude_edge):
    """Return max shortest path distance, or -1 if disconnected."""
    ex_u, ex_v = exclude_edge
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            nd = dist[u] + 1
            for v in adj[u]:
                if (u == ex_u and v == ex_v) or (u == ex_v and v == ex_u):
                    continue
                if dist[v] == -1:
                    dist[v] = nd
                    queue.append(v)
        for t in range(n):
            if dist[t] == -1:
                return -1
            if dist[t] > 2:
                return dist[t]
    return 2

def construct():
    """
    Return a list of edges [u, v] for a graph on vertices 0..15.
    The graph must have diameter exactly 2, and every edge must be critical
    (deleting it increases diameter above 2 or disconnects the graph).
    
    Key insight: For an edge (u,v) to be critical in a diameter-2 graph,
    there must be no common neighbor w of u and v (otherwise removing (u,v)
    still leaves dist(u,v) <= 2 via w). This means the graph must be 
    triangle-free.
    
    By Mantel's theorem, a triangle-free graph on n vertices has at most
    floor(n^2/4) edges. For n=16, this is 64 edges, achieved by K_{8,8}.
    
    K_{8,8} has diameter 2 (any two vertices in the same partition have
    distance 2 via any vertex in the other partition; vertices in different
    partitions have distance 1).
    
    Every edge in K_{8,8} is critical: removing edge (a, b) where a is in
    partition A and b is in partition B means a and b have no direct edge.
    Since a's neighbors are all in B (except b) and b's neighbors are all 
    in A (except a), there is no common neighbor. Thus dist(a,b) becomes 
    infinity (disconnected in terms of the path between them specifically,
    or more precisely, they are no longer connected at all in the sense that
    no path exists between them).
    
    Wait, let me reconsider. In K_{8,8}, if I remove edge (a,b), can a reach b?
    a is connected to all other vertices in B. b is connected to all other 
    vertices in A. Is there a path from a to b?
    a -> b' (where b' is in B, b' != b) -> ... but b' is only connected to A.
    b' -> a' (where a' is in A, a' != a) -> ... a' is connected to B.
    a' -> b? No, because (a', b) is an edge in K_{8,8}, so yes!
    
    Let me trace: a -> b' -> a' -> b. This is a path of length 3.
    So removing (a,b) does not disconnect the graph, but it increases the
    distance between a and b from 1 to 3. Thus the diameter increases to 3.
    
    So K_{8,8} is valid and has 64 edges, which is the maximum possible.
    """
    n = 16
    half = n // 2
    
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    
    return edges
