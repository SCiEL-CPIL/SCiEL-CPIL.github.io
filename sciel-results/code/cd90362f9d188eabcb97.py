"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _moore_n(k, g):
    """Moore bound: minimum n for k-regular girth g."""
    if g % 2 == 1:
        r = (g - 1) // 2
        return k * (k - 1) ** r
    else:
        r = g // 2
        return (k - 1) * (k - 1) ** r

def _build_from_sigma(k, sigma):
    """Build k-regular graph from permutation sigma on {0..k-1}.
    Vertices 0..n-1 where n = k * (k-1)^(g/2 - 1) or similar.
    Actually use the standard circulant-permutation construction:
    n = k * m where m = (k-1)^(g//2 - 1) for even g, or (k-1)^((g-1)//2) for odd g.
    Vertex (a, b) has neighbors:
      - (sigma(a), b+1 mod m) for one direction
      - (sigma_inv(a), b-1 mod m) for the other
    This gives a 2-regular structure... not quite k-regular.
    
    Let me use a different standard construction: the "Moore graph from permutation" approach.
    For k-regular girth g, we need n >= Moore bound.
    Use the layered construction:
    n = k + k*(k-1) + k*(k-1)^2 + ... 
    This is the BFS tree of depth (g-1)//2 from a root, plus the last level closed.
    
    Simpler: use the standard construction where vertices are (level, parent, index)
    and edges connect level i to level i+1.
    """
    # Use the permutation-based Moore candidate:
    # n = k * (k-1)^(g//2 - 1) for even g
    # n = k * (k-1)^((g-1)//2) for odd g  -- actually these differ
    # Let's just use n = Moore bound and try to find a valid embedding.
    # 
    # Alternative: use the "shift" construction.
    # For k-regular, girth g: n = k*(k-1)^(g/2-1) (even g) or k*(k-1)^((g-1)/2) (odd g)
    # Vertices: 0..n-1. Edge from v to (v + step) mod n? No, that's circulant.
    #
    # Let me use a concrete known family: the "generalized Petersen" doesn't work for large k.
    #
    # Best approach: use the standard BFS-tree closing.
    # Level 0: 1 vertex (root)
    # Level 1: k vertices
    # Level 2: k*(k-1) vertices
    # ...
    # Level L: k*(k-1)^(L-1) vertices where L = (g-1)//2
    # Total without closing: 1 + k + k*(k-1) + ... + k*(k-1)^(L-1)
    # For the Moore graph, the last level vertices connect back to level 0 or 1.
    #
    # This is complex. Let me use a simpler but valid approach:
    # Construct a random k-regular graph on n = Moore_bound vertices,
    # check girth, and if it fails, try again. For small enough n this works.
    # But n can be large (e.g., k=4, g=10: Moore bound = 4*3^4 = 324).
    #
    # Actually, let me use the permutation construction properly.
    # Standard: for k-regular girth 2r, n = k*(k-1)^(r-1).
    # Vertices are (i, j) where i in {0..k-1}, j in {0..(k-1)^(r-1)-1}.
    # Neighbors of (i, j):
    #   For the "tree" edges: (sigma_i(j), j') where the mapping depends on level.
    #
    # This is getting complicated. Let me use a direct random regular graph approach
    # with girth checking, which is simpler and can find valid graphs.

    pass  # placeholder, will be replaced

def _girth_check(n, edges, k, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Verify degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Check for short cycles via non-backtracking walks
    # For each vertex, do BFS tracking (current, parent) pairs up to depth (g-1)//2
    max_depth = (g - 1) // 2
    for start in range(n):
        # visited: dict from (node, prev) to distance
        visited = {(start, -1): 0}
        queue = [(start, -1, 0)]
        found_short = False
        while queue:
            node, prev, dist = queue.pop(0)
            if dist >= max_depth:
                continue
            for nbr in adj[node]:
                if nbr == prev:
                    continue
                new_key = (nbr, node)
                if new_key in visited:
                    # Cycle found: dist + 1 + visited[new_key]
                    cycle_len = dist + 1 + visited[new_key]
                    if cycle_len < g:
                        return False
                else:
                    visited[new_key] = dist + 1
                    queue.append((nbr, node, dist + 1))
    return True

def _random_k_regular(n, k, rng):
    """Generate a random k-regular simple graph on n vertices using configuration model + dedup."""
    if n * k < 2 or n * k % 2 != 0:
        return None
    if k >= n:
        return None
    # Use the swap-based method: start with random pairing, then fix multi-edges/loops
    half = n * k // 2
    # Build half-edge list
    halves = list(range(n)) * k
    rng.shuffle(halves)
    # Pair them up
    pairs = [(halves[2*i], halves[2*i+1]) for i in range(half)]
    # Convert to edge set, removing loops and duplicates
    edges = set()
    valid = True
    for u, v in pairs:
        if u == v:
            continue  # loop, skip (will re-pair later ideally, but for simplicity)
        a, b = min(u, v), max(u, v)
        if (a, b) in edges:
            continue
        edges.add((a, b))
    # This won't give exactly k-regular. Use a different approach:
    # Start with a random k-regular graph using the "swap" method.
    # Simpler: use networkx-style but we don't have it. Use the following:
    # Generate by starting with a known k-regular graph and swapping.
    # For now, use a greedy approach:
    # Each vertex needs k neighbors. Assign them one by one.
    adj = [set() for _ in range(n)]
    for v in range(n):
        available = [u for u in range(n) if u != v and u not in adj[v] and len(adj[u]) < k]
        if len(available) < k:
            return None
        chosen = rng.sample(available, k)
        for u in chosen:
            adj[v].add(u)
            adj[u].add(v)
    # Check if all degrees are exactly k
    for v in range(n):
        if len(adj[v]) != k:
            return None
    # Convert to edge list
    edge_set = set()
    for v in range(n):
        for u in adj[v]:
            if v < u:
                edge_set.add((v, u))
    return list(edge_set)

def construct(k, g):
    deadline = time.time() + 145.0
    rng = random.Random(42)
    
    # Moore bound
    n_moore = _moore_n(k, g)
    
    # Try increasing n from Moore bound upward
    # For each n, try to find a k-regular graph with girth >= g
    # Use random generation + girth check
    
    # Start with a reasonable n
    # For k=4, g=10: n_moore = 4*3^4 = 324
    # Known best: 320 (less than Moore bound? No, Moore bound is a lower bound)
    # Wait: n(k,g) >= Moore bound. For k=4, g=10: 1 + k + k(k-1) + k(k-1)^2 + k(k-1)^3 = 1+4+12+36+108 = 161? 
    # No, the Moore bound formula: for even g=2r: n >= (k-1)^r + (k-1)^(r-1) + ... 
    # Actually: n >= 2 * sum_{i=0}^{r-1} (k-1)^i for even g
    # For odd g=2r+1: n >= k * sum_{i=0}^{r-1} (k-1)^i
    
    # Let me recompute:
    if g % 2 == 0:
        r = g // 2
        n_moore = 2 * sum((k-1)**i for i in range(r))
    else:
        r = (g - 1) // 2
        n_moore = k * sum((k-1)**i for i in range(r))
    
    best_n = None
    best_edges = None
    
    # Try n values from n_moore upward
    for n in range(n_moore, n_moore + 200):
        if time.time() > deadline - 10:
            break
        if n * k % 2 != 0:
            continue
        # Try several random graphs
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            edges = _random_k_regular(n, k, rng)
            if edges is None:
                continue
            if _girth_check(n, edges, k, g):
                best_n = n
                best_edges = edges
                break
        if best_n is not None:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: return a valid but possibly larger graph
    # Use the original Lazebnik construction as fallback
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
    index, order, edges_list = {start: 0}, [start], []
    for vertex in order:
        if time.time() > deadline - 1:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_list.append((index[vertex], index[other]))
    return len(order), edges_list
# EVOLVE-BLOCK-END
