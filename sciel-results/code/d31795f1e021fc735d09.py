"""Small 4-regular graph of girth 10 (cage problem n(4,10)).

Strategy:
1. Try to find a 4-regular graph on 320 vertices with girth >= 10 using a
   fast girth-check heuristic (BFS from each vertex up to depth 4, checking
   for non-backtracking cycles of length < g).
   - Use random 4-regular graph generation via perfect matching of a 4-regular
     multigraph, then deduplicate / reject if multi-edges appear.
2. If not found within time budget, fall back to a known explicit construction
   (LUW / Lazebnik-Ustimenko-Woldar) which gives a valid 4-regular graph of
   girth >= 10 but with more vertices.

The key insight: the smallest known n(4,10) is 320. If we can find ANY 4-regular
graph on 320 vertices with girth >= 10, we tie the best known. Random 4-regular
graphs on 320 vertices have expected girth ~ O(log n) ~ 10-12, so there's a
reasonable chance one has girth >= 10.

For the fallback, I'll use a deterministic construction that's guaranteed valid.
"""

import time
import random
from collections import deque

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _gen_random_4_regular(n):
    """Generate a random simple 4-regular graph on n vertices using configuration model.
    
    Use the configuration model: each vertex has 4 half-edges, pair them randomly.
    Reject if multi-edges or loops appear.
    """
    if n < 4:
        return None
    
    # Each vertex gets 4 stubs
    stubs = []
    for v in range(n):
        for _ in range(4):
            stubs.append(v)
    
    # Random permutation to pair up stubs
    random.shuffle(stubs)
    
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None  # loop
        key = (min(u, v), max(u, v))
        if key in edges:
            return None  # multi-edge
        edges.add(key)
    
    return list(edges)

def _girth_check_fast(adj, n, g, max_depth=None):
    """Check if graph has girth >= g using BFS with non-backtracking constraint.
    
    For girth >= g, no cycle of length < g exists. We check for cycles of length
    4, 6, 8 (for g=10) by doing BFS to depth 4 from each vertex and checking
    if we can close a cycle.
    
    Actually, a simpler approach: for girth >= g, the graph must be (g/2)-geodetic,
    meaning between any two vertices there's at most one shortest path of length < g.
    
    Fast check: BFS from each vertex to depth floor((g-1)/2). If we find two
    different paths of the same length < g between the same pair of vertices,
    there's a cycle of length < g.
    
    Even faster: check for 4-cycles, 6-cycles, 8-cycles explicitly.
    """
    if max_depth is None:
        max_depth = (g - 1) // 2  # For g=10, check up to depth 4
    
    # Check for cycles of length 4
    # A 4-cycle: u-v-w-x-u where u,v,w,x distinct
    # For each edge (u,v), check if there's a common neighbor of u and v other than each other
    # Actually: for each pair of vertices (u,v), count common neighbors. If >= 2, there's a 4-cycle.
    # But this is O(n^2 * degree) which for n=320 is fine.
    
    # Better: use the fact that in a 4-regular graph, number of 4-cycles can be computed.
    # For each vertex u, for each pair of neighbors (v, w), if v and w have a common neighbor x != u,
    # then u-v-x-w-u is a 4-cycle.
    
    # Let's do BFS-based check up to depth 4
    # From each vertex, do BFS to depth 4. Track (vertex, path_length, parent).
    # If we reach a vertex at depth d that was already reached at depth d' where d + d' + 1 < g
    # via a different parent, we have a cycle.
    
    # Actually the simplest correct check:
    # For each vertex s, do BFS to depth max_depth.
    # Record dist[s][v] and parent[s][v].
    # If during BFS we find an edge (u, v) where dist[u] + dist[v] + 1 < g and
    # u != parent[s][v] and v != parent[s][u], then there's a cycle of length dist[u]+dist[v]+1.
    
    # For g=10, we need dist[u] + dist[v] + 1 >= 10, i.e., dist[u] + dist[v] >= 9.
    # Since max depth is 4, dist[u] + dist[v] <= 8, so this check only catches cycles
    # where both u and v are at depth <= 4 from s.
    
    # Hmm, this might miss some cycles. Let me think more carefully.
    
    # A cycle of length c < g exists iff there's a vertex s and two distinct
    # non-backtracking walks from s of length < c that meet.
    
    # Simplest correct approach for small g: explicitly search for cycles of length 4, 6, 8.
    
    # 4-cycles: for each vertex u, for each pair of neighbors v, w of u,
    # if v and w are adjacent, that's a triangle (cycle of 3). If v and w have
    # a common neighbor x != u, that's a 4-cycle u-v-x-w-u.
    
    # Let's just do: for each vertex, BFS to depth 4, and check for back-edges
    # that would create a cycle of length < g.
    
    adj_set = [set(a) for a in adj]
    
    for s in range(n):
        # BFS from s
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
                    if dist[u] + 1 > max_depth:
                        continue
                else:
                    # v already visited
                    if v != parent[u]:
                        # Cycle length: dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
            if dist[u] >= max_depth:
                continue
    
    return True

def _try_random_320(n, g, max_attempts=None):
    """Try to find a random 4-regular graph on n vertices with girth >= g."""
    if max_attempts is None:
        max_attempts = 10000
    
    for attempt in range(max_attempts):
        if _time_left() < 5.0:
            break
        
        edges = _gen_random_4_regular(n)
        if edges is None:
            continue
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Quick degree check
        if any(len(adj[i]) != 4 for i in range(n)):
            continue
        
        if _girth_check_fast(adj, n, g):
            return n, edges
    
    return None

def _luw_fallback(k, g):
    """Lazebnik-Ustimenko-Woldar fallback construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
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
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    
    # For k=4, g=10: try to find n=320
    if k == 4 and g == 10:
        # Try random graphs on 320 vertices
        result = _try_random_320(320, 10, max_attempts=50000)
        if result is not None:
            return result
        
        # Fallback to LUW
        return _luw_fallback(k, g)
    
    # Generic fallback
    return _luw_fallback(k, g)
