import time
import random
from collections import deque

_DEADLINE = 120.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _girth_violation(adj, g):
    """Return True iff there is a cycle of length < g.
    Uses non-backtracking BFS from every vertex; a cycle of length L is
    detected at depth L-1 (walk of L-1 non-backtracking steps returning to start).
    """
    n = len(adj)
    limit = g - 1  # we need to rule out walks of length <= g-2 returning to start
    # We run BFS for levels 1..limit. If at level L we return to the start,
    # that is a cycle of length L. We must find none for L in 1..g-2.
    max_level = g - 2
    if max_level <= 0:
        return False
    for s in range(n):
        # state: (node, prev)
        frontier = [(v, -1) for v in adj[s]]
        # visited per (node,prev) to avoid re-exploration
        seen = set(frontier)
        for _lvl in range(1, max_level + 1):
            if not frontier:
                break
            nxt = []
            for node, prev in frontier:
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == s:
                        # cycle of length _lvl
                        return True
                    st = (nb, node)
                    if st in seen:
                        continue
                    seen.add(st)
                    nxt.append(st)
            frontier = nxt
    return False

def _make_cubic(n, rng):
    """Build a simple 3-regular graph on n vertices via random 2-regular multigraph
    of edges. Returns adjacency list or None on failure."""
    m = n // 2
    # random 2-regular multigraph on m nodes: random 2-regular = union of cycles
    # Simplest: random permutation pairing trick -> random 2-regular multigraph
    # Use: assign each node two random neighbors (with replacement), then fix to be 2-regular
    # Easier: generate a random 2-regular graph by taking a random permutation of edges.
    # Method: random 2-regular multigraph = each vertex degree 2 in multigraph.
    # We'll do: start with a random permutation of [0..m-1] giving a 2-regular multigraph:
    #   edges (i, perm[i]) and (i, perm2[i]) where perm2 is inverse? 
    # Simpler robust approach: random 2-factor via random permutation.
    # A 2-regular multigraph can be built as: take random permutation p; edges (i, p[i])
    # forms disjoint cycles (each vertex degree 2 counting the cycle edges).
    perm = list(range(m))
    rng.shuffle(perm)
    # build edge multiset: for each i, edge (i, perm[i])
    edge_list = []
    for i in range(m):
        j = perm[i]
        if i == j:
            # self-loop in multigraph -> becomes a 2-cycle? Actually (i,i) means two parallel
            # edges between vertex-pair? In cubic construction, a self-loop in the 2-regular
            # multigraph maps to a 2-cycle in the cubic graph (two parallel edges) -> invalid.
            # We'll just reject and retry.
            return None
        edge_list.append((i, j))
    # Now build cubic graph: each multigraph vertex i -> two cubic vertices (2i, 2i+1)
    # Each multigraph edge (i,j) contributes two cubic edges: (2i,2j) and (2i+1,2j+1)
    adj = [[] for _ in range(n)]
    seen = set()
    for (i, j) in edge_list:
        for a, b in ((2 * i, 2 * j), (2 * i + 1, 2 * j + 1)):
            if a == b:
                return None
            key = (a, b) if a < b else (b, a)
            if key in seen:
                return None
            seen.add(key)
            adj[a].append(b)
            adj[b].append(a)
    # verify degree 3
    for v in range(n):
        if len(adj[v]) != 3:
            return None
    return adj

def construct(k, g):
    global _START
    _START = time.time()
    if k != 3:
        # fallback to a trivial valid construction if possible; for this task k=3
        # We still attempt a generic random regular graph.
        pass
    # Moore bound lower bound for cubic girth g:
    # n >= 2 * sum_{i=0}^{(g-2)/2} 2^i  (for even g)
    if g % 2 == 0:
        h = g // 2 - 1
        lower = 2 * sum(2 ** i for i in range(h + 1))
    else:
        h = (g - 1) // 2
        lower = 2 + 2 * sum(2 ** i for i in range(1, h + 1))
    lower = max(lower, 4)
    # n must be even
    if lower % 2:
        lower += 1

    rng = random.Random(12345)
    # Try increasing sizes. Start near a size expected to have girth >= g.
    # Girth of random cubic grows ~ log n, so n ~ 2^g is safe.
    # We'll start at a reasonable n and increase.
    sizes = []
    # include lower bound and a few geometric candidates
    n0 = max(lower, 64)
    # build a list of candidate n (even, increasing)
    # We'll just iterate: start at n0, then multiply by ~1.5
    n = n0
    best = None
    while _remaining() > 2.0:
        if n < lower:
            n = lower
        # ensure even
        if n % 2:
            n += 1
        # attempt a few random graphs at this size
        attempts = 0
        max_attempts = 40
        while attempts < max_attempts and _remaining() > 2.0:
            adj = _make_cubic(n, rng)
            attempts += 1
            if adj is None:
                continue
            if not _girth_violation(adj, g):
                # valid
                edges = []
                seen = set()
                for u in range(n):
                    for v in adj[u]:
                        key = (u, v) if u < v else (v, u)
                        if key not in seen:
                            seen.add(key)
                            edges.append((u, v))
                return n, edges
        # grow
        n = int(n * 1.4) + 2
        if n % 2:
            n += 1
    # fallback: return a guaranteed valid large graph if we have time; else best effort.
    # Construct a large random cubic graph with enough size to be safe.
    n = max(2 ** g, 256)
    if n % 2:
        n += 1
    while _remaining() > 0.5:
        adj = _make_cubic(n, rng)
        if adj is not None and not _girth_violation(adj, g):
            edges = []
            seen = set()
            for u in range(n):
                for v in adj[u]:
                    key = (u, v) if u < v else (v, u)
                    if key not in seen:
                        seen.add(key)
                        edges.append((u, v))
            return n, edges
        n = int(n * 1.5) + 2
    # last resort: return a trivial valid graph (a cycle won't be cubic). 
    # For safety, return the LUW-free simple construction: a 3-regular graph that is valid.
    # Use a large girth-16 known small graph? We'll just return a big random cubic and hope.
    # If we truly run out, return a minimal valid cubic graph with girth >= g is impossible
    # for small n; but evaluator expects valid. As a last resort, build a 3-regular graph
    # on 2^g vertices guaranteed girth >= g+? Not guaranteed. We'll return best found or None-ish.
    # To avoid crash, return a cycle-based cubic? Not cubic. We'll return a known valid:
    # the complete bipartite K3,3 has girth 4, invalid. 
    # We'll just return a large graph and trust earlier success.
    raise RuntimeError("could not find valid graph in time")
