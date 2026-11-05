"""Search for a small 5-regular graph with girth >= g.

We generate random k-regular graphs and verify their girth exactly via BFS
from each vertex up to depth (g-1)//2, keeping the smallest verified one.
A deadline guard returns the best valid result before the wall-clock limit.
"""
import random
import time
import numpy as np
from array import array

# ---------------- deterministic RNG ----------------
def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

# ---------------- k-regular graph generation ----------------
def _pairing_regular(n, k, rng):
    """Return adjacency (list of lists) of a random simple k-regular graph on n
    vertices via a random perfect matching of n*k stubs.  Returns None on
    failure (self-loop or parallel edge)."""
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    # random perfect matching
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [None] * n
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if seen[u] is not None:
            if v in seen[u]:
                return None
            seen[u].append(v)
        else:
            seen[u] = [v]
        adj[u].append(v)
        if seen[v] is not None:
            seen[v].append(u)
        else:
            seen[v] = [u]
        adj[v].append(u)
    return adj

def _swap_regular(adj, adj_set, n, k, rng, stubs=None):
    """One random 2-switch preserving k-regularity; returns True if it produced
    a simple graph (no self-loops / parallel edges)."""
    m = n * k
    if stubs is None:
        stubs = array('I', (i for i in range(n) for _ in range(k)))
    a = _randrange(rng, m)
    b = _randrange(rng, m - 1)
    if b >= a:
        b += 1
    c = _randrange(rng, m - 2)
    if c >= a:
        c += 1
    if c >= b:
        c += 1
    d = _randrange(rng, m - 3)
    for x in (a, b, c):
        if d >= x:
            d += 1
    u1, v1 = stubs[a], stubs[b]
    u2, v2 = stubs[c], stubs[d]
    # remove edges (u1,v1) and (u2,v2); require they are distinct
    if u1 == v1 or u2 == v2:
        return False
    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
        return False
    # Remove from both list and set
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj_set[u1].discard(v1)
    adj_set[v1].discard(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    adj_set[u2].discard(v2)
    adj_set[v2].discard(u2)
    # add edges (u1,u2) and (v1,v2); forbid self-loops / parallel edges
    if u1 == u2 or v1 == v2:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj_set[u1].add(v1)
        adj_set[v1].add(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        adj_set[u2].add(v2)
        adj_set[v2].add(u2)
        return False
    if u2 in adj_set[u1] or v2 in adj_set[v1]:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj_set[u1].add(v1)
        adj_set[v1].add(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        adj_set[u2].add(v2)
        adj_set[v2].add(u2)
        return False
    adj[u1].append(u2)
    adj[u2].append(u1)
    adj_set[u1].add(u2)
    adj_set[u2].add(u1)
    adj[v1].append(v2)
    adj[v2].append(v1)
    adj_set[v1].add(v2)
    adj_set[v2].add(v1)
    return True

# ---------------- exact girth check: does graph have a cycle shorter than g? ----------------
def _girth_ok(adj, g):
    """Return True if the graph has no cycle shorter than g.
    Uses BFS from each vertex up to depth (g-1)//2. If any vertex is reached
    from two different neighbors at the same depth, or a vertex is reached at
    depth d where d + 1 + d >= g would form a short cycle, we detect it.
    
    Actually, simpler: BFS from each source s up to depth floor((g-1)/2).
    If during BFS from s we reach s again (cycle), or if two different paths
    of length <= floor((g-1)/2) from s meet at the same vertex via different
    parents, that indicates a cycle of length <= 2*floor((g-1)/2) + 1 or
    2*floor((g-1)/2) < g.
    
    Most direct: for each vertex s, do BFS tracking parent. If we find an
    edge (u,v) where dist[u] + dist[v] + 1 < g and u != parent of v (and
    v != parent of u), we have a cycle shorter than g.
    """
    n = len(adj)
    if n == 0:
        return True
    
    max_depth = (g - 1) // 2
    if max_depth < 1:
        return True
    
    # For each source, BFS up to max_depth, tracking distance and parent
    # A cycle of length L exists iff there's an edge (u,v) with
    # dist_s[u] + dist_s[v] + 1 <= L for some source s, where the paths
    # don't share the edge (u,v) itself.
    # Simpler approach: BFS from each s, if we encounter an edge (u,v) where
    # both are at distance d from s and d + d + 1 < g, we have a short cycle.
    # Or if one is at distance d and the other at d' and d + d' + 1 < g.
    
    # Even simpler and exact: just do BFS from each vertex s up to depth
    # (g-1)//2. If we can reach s itself at distance < g, there's a cycle.
    # But BFS doesn't naturally return to s. Instead: for each edge (u,v),
    # check if there's a path from u to v of length < g-1 that doesn't use
    # edge (u,v). This is equivalent to: in the graph with edge (u,v) removed,
    # is the distance from u to v less than g-1?
    
    # Most efficient for small g: for each vertex s, BFS up to depth (g-1)//2.
    # Record for each vertex v the set of "directions" (neighbors of v that
    # are at distance one less from s). If two different directions reach v
    # at the same distance d, then there's a cycle of length 2*d.
    # If a direction reaches v at distance d and another at d', cycle length
    # is d + d' + 1 (via the edge between the two predecessors).
    
    # Let me just do: for each s, BFS up to depth max_depth. For each vertex
    # visited, store which neighbor of s (or more generally, the "parent" 
    # chain) led there. If two different parents lead to the same vertex at
    # depths d1, d2, we have a cycle of length d1 + d2 + 1 (if the edge 
    # connects the parents) or d1 + d2 (if we go back through s).
    
    # Cleanest correct approach: 
    # For each vertex s, do BFS. dist[v] = shortest distance from s to v.
    # For each edge (u,v) where u != s and v != s:
    #   if dist[u] + 1 + dist[v] < g and the shortest paths don't share edge (u,v):
    #     Actually, dist[u] + dist[v] + 1 gives a cycle length if the paths
    #     from s to u and s to v are internally disjoint and don't use edge (u,v).
    #     A sufficient condition: if dist[u] + dist[v] + 1 < g, there EXISTS
    #     a cycle of that length or shorter.
    
    # This is correct: for any edge (u,v), if dist_s[u] + dist_s[v] + 1 < g,
    # then there is a cycle of length at most dist_s[u] + dist_s[v] + 1 < g.
    # Proof: take shortest path from s to u (length dist_s[u]), edge (u,v),
    # shortest path from v to s (length dist_s[v]). This walk has length
    # dist_s[u] + 1 + dist_s[v]. It may have repeated vertices, but it 
    # contains a cycle of length at most that value.
    
    # So: graph has girth >= g iff for ALL vertices s and ALL edges (u,v) 
    # with u != s, v != s: dist_s[u] + dist_s[v] + 1 >= g.
    # Equivalently: for all s and all edges (u,v) not incident to s:
    #   dist_s[u] + dist_s[v] >= g - 1.
    
    # We can check this more efficiently: for each s, BFS up to depth
    # (g-1)//2. For each vertex v at distance d from s, for each neighbor w
    # of v (w != s), if dist_s[w] < g - 1 - d, we have a violation.
    # Actually just: for each edge (u,v) where both u,v are within distance
    # (g-1)//2 from s, check dist_s[u] + dist_s[v] + 1 >= g.
    
    # Let me just implement the straightforward check:
    for s in range(n):
        # BFS from s up to max_depth
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            du = dist[u]
            if du >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    queue.append(v)
        
        # Check all edges: for each edge (u,v), if both are reachable from s
        # and dist[u] + dist[v] + 1 < g, we have a short cycle.
        # But we only need to check edges where at least one endpoint is not s.
        for u in range(n):
            if dist[u] == -1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    continue
                if u == s or v == s:
                    # Edge incident to s: cycle length is 1 + dist[v] (if v != s)
                    # Actually if u == s, cycle is 1 + dist[v] only if there's
                    # a path from v back to s not using edge (s,v). dist[v] is
                    # the shortest path, which might use edge (s,v). 
                    # Skip edges incident to s for this check; they're handled
                    # by the general condition when we consider other sources.
                    continue
                if dist[u] + dist[v] + 1 < g:
                    return False
    
    return True

# ---------------- main entry point ----------------
def construct(k, g):
    deadline = time.time() + 110.0   # safety margin under the 150 s limit
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    def consider(n, adj):
        nonlocal best, best_n
        if _girth_ok(adj, g):
            best = adj
            best_n = n
            return True
        return False

    # (1) try sizes where girth >= g is plausible
    # For 5-regular graphs, girth 9 requires roughly n >= 100+
    # Start at reasonable sizes and go up
    if k == 5 and g == 9:
        start_n = 100
    else:
        start_n = 2 * k

    n_candidates = []
    n = start_n
    while n <= 600 and time.time() < deadline:
        # Ensure n*k is even
        if (n * k) % 2 != 0:
            n += 1
            continue
        n_candidates.append(n)
        n += 2  # step by 2 to keep n*k even when k is odd

    for n in n_candidates:
        if time.time() > deadline:
            break
        adj = None
        for _ in range(200):
            if time.time() > deadline:
                break
            cand = _pairing_regular(n, k, rng)
            if cand is not None:
                adj = cand
                break
        if adj is not None:
            if consider(n, adj):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
            # Try 2-switch local search
            adj_set = [set(adj[i]) for i in range(n)]
            stubs = array('I', (i for i in range(n) for _ in range(k)))
            check_interval = 100
            for it in range(5000):
                if time.time() > deadline:
                    break
                if _swap_regular(adj, adj_set, n, k, rng, stubs):
                    if (it + 1) % check_interval == 0:
                        if consider(n, adj):
                            return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
            # Final check
            if consider(n, adj):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

    if best is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best[i] if i < j]

    # (3) guaranteed fallback: the Lazebnik-Ustimenko-Woldar construction
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
