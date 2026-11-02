"""New program for the suite task; uses Boutin's recursive lifting construction.

Approach (different family from the seed's L-U-W algebraic construction):
  Boutin's construction: from a k-regular graph H of girth h, build a k-regular
  graph G of girth >= 2h - 2 by replacing every vertex of H with a k-cycle and
  wiring the cycles together according to H's edges. We start from a small
  k-regular base cage of girth h = ceil((g+2)/2) and lift once.
"""
# EVOLVE-BLOCK-START
import random

def _build_base_cage(k, h, rng):
    """Build a k-regular simple graph of girth at least h via greedy edge
    insertion with a BFS-based girth check, then local repair. Returns (n, edges).
    Uses as few vertices as possible consistent with the Moore bound."""
    # Moore bound for k-regular girth h:
    if h % 2 == 1:
        lb = 1 + k * (k ** ((h - 1) // 2) - 1) // (k - 1)
    else:
        lb = 2 * k * (k ** ((h - 2) // 2) - 1) // (k - 1)
    n = max(lb, 2 * k)
    # grow n until we can find a valid graph
    while True:
        ok = _try_greedy(k, h, n, rng)
        if ok is not None:
            return n, ok
        n += max(1, k)
        if n > 20000:
            raise RuntimeError("base cage search failed")

def _try_greedy(k, h, n, rng):
    """Attempt to build a k-regular girth->=h graph on n vertices greedily."""
    adj = [set() for _ in range(n)]
    deg = [0] * n
    # candidate edges: all non-adjacent, non-self pairs, shuffled
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    rng.shuffle(pairs)
    # We need each vertex to reach degree k. Greedy: add an edge if it doesn't
    # create a cycle shorter than h.
    # Track distance-2 (and up to (h-1)/2) neighborhoods for girth check.
    # Simpler: after each insertion, verify no short cycle via bounded BFS.
    def creates_short_cycle(u, v):
        # check whether u and v are connected by a path of length < h-1
        # (adding edge u-v then closes a cycle of length <= h-1)
        limit = h - 2
        # BFS from u up to depth limit, see if we reach v
        seen = {u: 0}
        frontier = [u]
        depth = 0
        while frontier and depth < limit:
            nxt = []
            for x in frontier:
                for y in adj[x]:
                    if y == v:
                        return True
                    if y not in seen:
                        seen[y] = depth + 1
                        nxt.append(y)
            frontier = nxt
            depth += 1
        return False

    # order of vertex additions: prefer lowest degree first
    for _ in range(n * k // 2):
        # pick a vertex with degree < k
        low = [i for i in range(n) if deg[i] < k]
        if not low:
            break
        low.sort(key=lambda i: deg[i])
        u = low[0]
        # find a candidate partner
        cand = [j for j in range(n) if j != u and j not in adj[u] and deg[j] < k]
        rng.shuffle(cand)
        placed = False
        for v in cand:
            if creates_short_cycle(u, v):
                continue
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            placed = True
            break
        if not placed:
            return None
    if any(d != k for d in deg):
        return None
    # final girth verification
    if _min_girth(adj, n) < h:
        return None
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return edges

def _min_girth(adj, n):
    """Exact minimum cycle length via BFS from each vertex (non-backtracking)."""
    best = 10 ** 9
    for s in range(n):
        dist = [-1] * n
        par = [-1] * n
        dist[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]; head += 1
            for v in adj[u]:
                if v == par[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    q.append(v)
                else:
                    cyc = dist[u] + dist[v] + 1
                    if cyc < best:
                        best = cyc
    return best

def _boutin_lift(k, h, n, edges):
    """Boutin lift: replace each vertex i of the base graph by a k-cycle
    (i,0),(i,1),...,(i,k-1). For each base edge (i,j), connect the cycles.
    Result is k-regular with girth >= 2h - 2."""
    # new vertex id: i*k + t
    N = n * k
    adj = [set() for _ in range(N)]
    # intra-cycle edges
    for i in range(n):
        for t in range(k):
            a = i * k + t
            b = i * k + ((t + 1) % k)
            adj[a].add(b)
            adj[b].add(a)
    # inter-cycle edges according to base graph
    # For each base vertex i, its base-neighbors are neighbors[i] (k of them).
    # Connect (i,t) to (neighbor_t, t) for t in 0..k-1, where neighbor_t is the
    # t-th neighbor of i (in a fixed order).
    nbrs = [[] for _ in range(n)]
    for (i, j) in edges:
        nbrs[i].append(j)
        nbrs[j].append(i)
    for i in range(n):
        assert len(nbrs[i]) == k, (i, len(nbrs[i]))
        for t in range(k):
            j = nbrs[i][t]
            a = i * k + t
            b = j * k + t
            adj[a].add(b)
            adj[b].add(a)
    out_edges = []
    for i in range(N):
        for j in adj[i]:
            if i < j:
                out_edges.append((i, j))
    return N, out_edges

def construct(k, g):
    # base girth needed: 2h - 2 >= g  =>  h >= (g+2)/2
    h = (g + 2) // 2 + (1 if (g + 2) % 2 else 0)
    # ensure h >= 3
    h = max(3, h)
    rng = random.Random(12345)
    base_n, base_edges = _build_base_cage(k, h, rng)
    n, edges = _boutin_lift(k, h, base_n, base_edges)
    return n, edges
# EVOLVE-BLOCK-END
