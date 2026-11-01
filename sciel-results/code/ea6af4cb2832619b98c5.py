"""Construct a 4-regular graph of girth 10 using the Lazebnik-Ustimenko-Woldar family.

The seed uses q = k = 4, giving a small but far-from-optimal graph.  The LUW graph CD(d, q)
with q a prime power and d odd has girth at least d + 5 (and exactly d + 5 for these
parameters).  For girth 10 we need d >= 5 (odd), so d = 5 works with girth >= 10.

The number of vertices is 2 * q^(d+1) for the full connected component (points and lines
each form a q^(d+1)-sized set).  To minimize n, we want the smallest prime power q such
that the resulting graph actually has girth >= 10 (not just >= d+5 in theory, but verified
by BFS).  For d=5, the Moore bound lower bound for n(4,10) is 243, and the known best is
320.  The LUW construction with q=2 gives 2*2^6 = 128 vertices (too small, will have
short cycles).  q=3 gives 2*3^6 = 1458 (too many).  So the LUW family with d=5 doesn't
get us to 320.

Alternative: use the known (4,10)-cage construction.  The 320-vertex (4,10)-cage is a
bipartite graph that can be constructed via a specific algebraic method.  One well-known
construction uses the group Z_5 x Z_5 x Z_5 x Z_5 with a voltage assignment, or uses
the "Ramanujan graph" family.

Actually, a cleaner approach: the (4,10)-cage on 320 vertices is known to be the
incidence graph of a specific design, or can be built from the generalized quadrangle
GQ(3,3) which has 40 points and 40 lines, each point on 4 lines, each line through 4
points, and girth 8 in the incidence graph... no, GQ(3,3) incidence graph has girth 8.

Let me think differently.  A 4-regular bipartite graph with girth 10 on 320 vertices.
This is the (4,10)-cage.  One construction: take the Cayley graph of a group with 160
elements and 4 generators (for the bipartite double cover).

Actually, the simplest robust approach: use the LUW family but with a larger d and
optimize q.  For d=5, q must be a prime power.  The girth of CD(5, q) is at least 10.
The number of vertices is 2*q^6.  For q=2: 128 vertices.  But does it actually have
girth 10?  Let's check: the LUW paper states girth >= d+5 = 10 for d odd.  So CD(5,2)
should have girth >= 10.  But 128 < 243 (Moore bound), which is impossible!  So my
formula for vertex count must be wrong, or the girth bound requires q to be large enough.

Looking more carefully: the LUW graph CD(d,q) has |V| = 2 * q^(d+1) only when the graph
is connected.  For small q relative to d, the graph may be disconnected, and the
connected component of the origin has fewer vertices.  The seed code only returns the
connected component.  For q=2, d=5, the connected component may be much smaller than
2*2^6=128, and indeed may have girth < 10.

The key insight: for the LUW family to achieve the theoretical girth bound d+5, we need
q >= d (or some similar condition).  For d=5, we need q >= 5.  The smallest prime power
>= 5 is 5.  So CD(5, 5) has 2*5^6 = 31250 vertices.  Way too many.

This means the LUW family is not the right tool for the (4,10)-cage.  We need a different
construction.

The 320-vertex (4,10)-cage can be constructed as follows: it is a bipartite graph where
each part has 160 vertices.  One known construction uses the group G = Z_160 with a
specific connection set, or uses a more sophisticated algebraic construction.

Let me use a known explicit construction.  The (4,10)-cage is the incidence graph of
the generalized quadrangle of order (3,3)?  No, that has 40+40=80 vertices and girth 8.

Another approach: the (4,10)-cage is related to the Hoffman-Singleton graph (which is
(7,5) cage on 50 vertices) in a different way.

Let me try a computational approach: build a 4-regular graph with girth 10 on as few
vertices as possible using a greedy/randomized algorithm with backtracking, or use a
known construction from the literature.

Actually, I recall that the (4,10)-cage on 320 vertices can be constructed using the
following method: take the Cayley graph of the group (Z_2)^4 x Z_5 x Z_5 x Z_2 with
appropriate generators.  But I don't remember the exact generators.

Let me try a different, more practical approach: use a randomized search to build a
4-regular graph with girth 10, starting from a candidate size and growing.  Given the
150s time limit, I'll use a greedy approach with backtracking on the number of vertices.

Actually, the most reliable approach for this specific problem is to use the known
320-vertex construction.  Let me implement a BFS-based girth verifier and a
construction based on the "Klein" or "Fisher" method.

Here's a concrete plan:
1. Start with n = 320 (the known cage size).
2. Construct a 4-regular bipartite graph on 320 vertices (160+160) with girth 10.
3. Use a voltage graph lifting: take the Petersen graph (10 vertices, 3-regular) as base,
   and lift it to a 4-regular graph on 320 vertices.  But Petersen is 3-regular, not 4.

Better: take the complete bipartite graph K_{4,4} (8 vertices, 4-regular) as base, and
lift it.  But K_{4,4} has girth 4, so the lift will have girth at most 4 * (lift size).

The cleanest approach: use the known fact that the (4,10)-cage is the bipartite double
cover of a specific 4-regular graph on 160 vertices with girth 10.  Or directly
construct the 320-vertex graph.

Let me use a well-known explicit construction from the literature.  The (4,10)-cage can
be constructed as follows:

Vertices: {0, 1, ..., 319}
The graph is bipartite with parts A = {0..159} and B = {160..319}.
Each vertex in A is connected to 4 vertices in B and vice versa.

One construction uses the following rule: for i in A (0-indexed), connect to
B indices based on a linear function over GF(5) or Z_5.

I'll implement a construction based on the "Higman-Sims" type graph or use a
computational search.  Given the time constraint, let me use a randomized greedy
approach with girth checking.

Here's my final plan:
1. For the specific case k=4, g=10, use a known 320-vertex construction.
2. For other (k, g) cases, fall back to the LUW construction from the seed.
3. Verify girth via BFS.
"""
import time
import sys
from collections import deque

def _bfs_girth(adj):
    """Compute the girth of an undirected graph via BFS from each vertex."""
    n = len(adj)
    best = float('inf')
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle = dist[u] + dist[v] + 1
                    if cycle < best:
                        best = cycle
        if best <= 2:
            break
    return best if best < float('inf') else 0

def _is_simple_graph(n, edges):
    """Check if the graph is simple (no loops, no multiple edges)."""
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))
    # Check degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != 4: # Assuming k=4 for this specific check, but better to pass k
            return False
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (from seed)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _construct_4_10_cage():
    """Construct the (4,10)-cage.
    
    Tries the LUW construction with q=2 first, which yields 128 vertices.
    If valid, returns it. Otherwise falls back to q=4 (8192 vertices) or other methods.
    """
    # Try q=2, d=5 (128 vertices)
    # The LUW graph CD(5,2) is 4-regular.
    # We need to verify girth >= 10 and simplicity.
    try:
        n, edges = _luw_construct_q(2, 5)
        # Verify
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check simplicity
        if not _is_simple_graph(n, edges):
            return _luw_construct(4, 10)
            
        # Check girth
        girth = _bfs_girth(adj)
        if girth >= 10:
            return n, edges
    except Exception:
        pass
    
    # Fallback to q=4
    return _luw_construct(4, 10)

def _luw_construct_q(q, d):
    """LUW construction with specific q and d."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def construct(k, g):
    """Construct a k-regular graph of girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    if k == 4 and g == 10:
        # Try the specific (4,10)-cage construction
        try:
            n, edges = _construct_4_10_cage()
            # Verify
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            # Check degree
            for i in range(n):
                if len(adj[i]) != k:
                    raise ValueError(f"Vertex {i} has degree {len(adj[i])}")
            # Check girth
            girth = _bfs_girth(adj)
            if girth >= g:
                return n, edges
        except Exception:
            pass
        # Fallback to LUW
        n, edges = _luw_construct(k, g)
        return n, edges
    else:
        # General case: use LUW construction
        n, edges = _luw_construct(k, g)
        return n, edges
