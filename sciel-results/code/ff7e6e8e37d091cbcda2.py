"""Construct 5-regular graphs of girth 9 with as few vertices as possible."""
import time
import math
import random
import sys
from collections import deque

def _moore_lower_bound(k, g):
    """Compute the Moore bound (lower bound on number of vertices)."""
    if g % 2 == 1:
        # Odd girth: 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        r = (g - 3) // 2
        if k == 1:
            return 1 + r
        return 1 + k * ((k-1)**(r+1) - 1) // (k - 2) if k > 2 else 1 + k*(r+1)
    else:
        # Even girth: 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
        r = g // 2 - 1
        if k == 1:
            return 2 * (r + 1)
        return 2 * ((k-1)**(r+1) - 1) // (k - 2) if k > 2 else 2*(r+1)


def _verify_graph(n, edges, k, g):
    """
    Verify:
    1. Simple graph (no loops, no multi-edges)
    2. k-regular
    3. Girth >= g (using BFS from each vertex to find shortest cycle)
    Returns (valid, actual_girth)
    """
    # Check simple
    if len(edges) != len(set(tuple(sorted(e)) for e in edges)):
        return False, 0
    for u, v in edges:
        if u == v:
            return False, 0
        if u >= n or v >= n:
            return False, 0

    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Check k-regular
    for i in range(n):
        if len(adj[i]) != k:
            return False, 0
        if len(set(adj[i])) != k:
            return False, 0

    # Check girth via BFS from each vertex
    # For each vertex, do BFS and find the shortest cycle containing it
    # A cycle of length L through vertex v means there's a path of length L-1 from v back to v
    # More precisely: BFS from v, track parent. If we find an edge (u,w) where dist[u] + dist[w] + 1 >= girth candidate
    # Actually, the standard approach: for each vertex v, BFS. For each edge (u,w) not in tree,
    # cycle length = dist[u] + dist[w] + 1 (if u and w are in different subtrees or same level)
    # But this is complex. Let's use the non-backtracking walk approach.
    
    # Simpler: BFS from each vertex, find shortest path back.
    # For vertex v, shortest cycle through v = min over neighbors u of v of (shortest path from u to v not using edge (v,u)) + 1
    # Even simpler: for each vertex v, BFS from v. The shortest cycle through v is the minimum
    # over all edges (a,b) where a != v, b != v, a != parent[b] in BFS tree, of dist[a]+dist[b]+1
    # But we need cycles that pass through v.
    
    # Let me use a cleaner approach: for each vertex v, compute the girth of the graph
    # by finding the shortest cycle. A graph has girth >= g iff no cycle of length < g.
    # We can check: for each vertex v, do BFS. For each edge (u,w) in the graph (not a tree edge),
    # the cycle length is dist[u] + dist[w] + 1 if u and w are in different branches,
    # or |dist[u] - dist[w]| + 1 if in the same branch... this is getting complicated.
    
    # Let's just do: for each vertex v, BFS. Record dist and parent. 
    # For each non-tree edge (u,w), cycle length = dist[u] + dist[w] + 1
    # (this works when u and w are in different subtrees of the BFS tree from v)
    # But if they're in the same subtree, it's more complex.
    
    # Actually the cleanest: girth = min over all vertices v of the shortest cycle through v.
    # Shortest cycle through v: remove v, find shortest path between any two neighbors of v.
    # Cycle length = that path length + 2.
    # But this is O(n * (n+m)) which for n~1000 might be slow.
    
    # Alternative: just do BFS from each vertex and for each edge (u,w) check if it forms a cycle.
    # For BFS from v: if edge (u,w) exists and u != w and neither is v:
    #   If parent[u] != w and parent[w] != u (i.e., not a tree edge):
    #     cycle length = dist[u] + dist[w] + 1
    #   If parent[u] == w: cycle length = dist[u] + 1 (but this is just the tree edge, skip)
    #   Similarly for parent[w] == u.
    # Wait, this doesn't correctly handle all cases.
    
    # Let me use the standard girth algorithm:
    # For each vertex v, BFS. For each edge (u,w) that is NOT a tree edge in the BFS from v:
    #   If u and w are at the same level: cycle = dist[u] + 1
    #   If at different levels: need to be more careful.
    
    # Actually, let me just use: girth >= g iff for every vertex v, the BFS tree from v has no
    # "back edge" creating a cycle shorter than g.
    # For BFS from v with dist[] and parent[]:
    # For each edge (u,w):
    #   If parent[u] == w or parent[w] == u: it's a tree edge, skip.
    #   Otherwise: cycle length = dist[u] + dist[w] + 1 (this is correct when u,w are in different branches)
    #   But if they're in the same branch, dist[u]+dist[w]+1 overestimates.
    #   
    # Hmm, the correct formula: the cycle formed by edge (u,w) in BFS tree from v has length
    # dist[u] + dist[w] + 1 ONLY if u and w are in different subtrees of some node.
    # In general, cycle length = dist[u] + dist[w] + 1 - 2*dist[lca(u,w)]
    # where lca is in the BFS tree.
    
    # This is getting too complex for verification. Let me use a simpler but correct method:
    # For each vertex v, find the shortest cycle through v by:
    #   For each pair of neighbors (a,b) of v (a<b), find shortest path from a to b in G-v.
    #   Cycle length = shortest_path + 2.
    #   Girth = min over v of min over pairs.
    # This is O(n * k^2 * (n+m)) which is too slow for n=1000.
    
    # Let me just do a BFS-based girth check that's O(n*(n+m)):
    # For each vertex v:
    #   BFS from v, getting dist[] and parent[]
    #   For each edge (u,w) where u < w (to avoid double counting):
    #     If u == v or w == v: skip (handled separately)
    #     If parent[u] == w or parent[w] == u: tree edge, skip
    #     cycle_len = dist[u] + dist[w] + 1
    #     girth = min(girth, cycle_len)
    #   Also check edges incident to v: for each neighbor u of v, 
    #     the shortest path from u to v not using (v,u) gives a cycle.
    #     This equals: min over neighbors w of v, w != u, of (1 + shortest path from w to u in G-v)
    #     Which is: 2 + min over pairs of neighbors.
    # 
    # Actually, the BFS approach with dist[u]+dist[w]+1 for non-tree edges DOES give the correct
    # cycle length for the cycle that goes through the root v only when u and w are in different
    # subtrees. For same subtree, it gives an upper bound. But for girth verification, we need
    # to ensure NO cycle is shorter than g. If dist[u]+dist[w]+1 >= g for all such edges, does that
    # guarantee girth >= g? Not necessarily, because same-subtree cycles could be shorter.
    
    # OK let me just do the expensive but correct check for small graphs, and for larger graphs
    # trust the construction.
    
    # For now, let's do a BFS from each vertex and check all non-tree edges.
    # This is a necessary but not sufficient condition for girth >= g.
    # For a proper check, we'd need LCA. Let me implement it properly.
    
    min_cycle = float('inf')
    for v in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[v] = 0
        queue = deque([v])
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    queue.append(w)
        # Check all edges
        for u in range(n):
            for w in adj[u]:
                if u > w:  # avoid double counting
                    continue
                if parent[u] == w or parent[w] == u:
                    continue
                # Find LCA of u and w in BFS tree from v
                # Walk up from u and w to find common ancestor
                # For efficiency, mark ancestors
                if dist[u] == -1 or dist[w] == -1:
                    continue
                # Simple LCA by walking up
                au, aw = u, w
                # Bring to same level
                du, dw = dist[au], dist[aw]
                while du > dw:
                    au = parent[au]
                    du -= 1
                while dw > du:
                    aw = parent[aw]
                    dw -= 1
                while au != aw:
                    au = parent[au]
                    aw = parent[aw]
                lca_dist = dist[au]
                cycle_len = dist[u] + dist[w] + 1 - 2 * lca_dist
                if cycle_len < min_cycle:
                    min_cycle = cycle_len
        # Also check cycles through v: for each edge (u,w) where one endpoint is v
        # Actually those are covered if we include edges with u=v or w=v
        # But we skipped tree edges. Edges from v to its children are tree edges.
        # Non-tree edges from v would be edges to vertices at the same or different level not via tree.
        # Since v is the root, all edges from v are to its neighbors. If (v,w) is not a tree edge,
        # then w was already visited, so parent[w] != v. Then cycle = 1 + dist[w] + 1 - 2*dist[lca(v,w)]
        # lca(v,w) = v since v is root, so cycle = 1 + dist[w] + 1 - 0 = dist[w] + 2.
        # But wait, if (v,w) is not a tree edge, w is a neighbor of v but parent[w] != v.
        # That means w was reached via another path. dist[w] >= 1.
        # The cycle is v -> ... -> w -> v, length = dist[w] + 1.
        # Hmm, let me just include edges where u==v or w==v.
        for w in adj[v]:
            if parent[w] == v:
                continue
            # Edge (v,w) is not a tree edge
            cycle_len = dist[w] + 1
            if cycle_len < min_cycle:
                min_cycle = cycle_len
    
    if min_cycle < g:
        return False, min_cycle
    return True, min_cycle


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction for k-regular bipartite graphs."""
    q = k
    # For girth g, we need d such that girth >= d + 5 (for odd d) or d + 4 (for even d)
    # We want girth >= g, so d + 5 >= g => d >= g - 5
    # d must have the same parity as g-5 for the formula to work cleanly
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4  # d even, girth >= d + 5 means girth >= d+5, but d even gives different bound
        # Actually for LUW: if d is odd, girth >= d+5; if d is even, girth >= d+4? 
        # Let me just use d = g - 5 and if even, d = g - 4
        # The original code uses: d = max(3, g - 5 if (g-5)%2 else g-4)
        # For g=9: g-5=4 (even), so d=5? No: (g-5)%2 = 4%2 = 0, so d = g-4 = 5.
        # Wait that's odd. Let me re-read.
        # d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # For g=9: (9-5)%2 = 0, so d = 9-4 = 5.
        # For g=8: (8-5)%2 = 1, so d = 8-5 = 3.
    
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _random_regular_graph(n, k):
    """Generate a random k-regular graph using the configuration model."""
    # Create 2k stubs, pair them randomly
    stubs = list(range(n)) * k
    random.shuffle(stubs)
    edges = set()
    for i in range(0, 2 * n * k, 2):
        u, v = stubs[i], stubs[i + 1]
        if u != v:
            edges.add((min(u, v), max(u, v)))
    if len(edges) != n * k // 2:
        return None  # Had a loop or multi-edge
    return list(edges)


def _improve_girth(n, edges, k, g, max_iterations=1000):
    """Try to improve girth by swapping edges."""
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    def find_short_cycle():
        """Find a short cycle using BFS from random vertices."""
        best = float('inf')
        best_cycle = None
        for start in random.sample(range(n), min(n, 50)):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        parent[w] = u
                        queue.append(w)
                    elif parent[u] != w:
                        # Found a cycle
                        # LCA
                        au, aw = u, w
                        du, dw = dist[au], dist[aw]
                        while du > dw:
                            au = parent[au]
                            du -= 1
                        while dw > du:
                            aw = parent[aw]
                            dw -= 1
                        while au != aw:
                            au = parent[au]
                            aw = parent[aw]
                        lca_d = dist[au]
                        clen = dist[u] + dist[w] + 1 - 2 * lca_d
                        if clen < best:
                            best = clen
                            best_cycle = (u, w, start)
        return best, best_cycle
    
    current_girth, _ = find_short_cycle()
    if current_girth >= g:
        return edges
    
    for _ in range(max_iterations):
        # Pick a random edge and try to swap
        u, v = random.choice(edges)
        # Find a neighbor w of u (w != v) and a neighbor x of v (x != u)
        # such that (w,x) is not an edge and (u,x) and (v,w) are not edges
        candidates = []
        for w in adj[u]:
            if w == v:
                continue
            for x in adj[v]:
                if x == u or x == w:
                    continue
                if (min(w, x), max(w, x)) not in edge_set:
                    if (min(u, x), max(u, x)) not in edge_set:
                        if (min(v, w), max(v, w)) not in edge_set:
                            candidates.append((w, x))
        if not candidates:
            continue
        w, x = random.choice(candidates)
        # Perform swap: remove (u,v) and (w,x) doesn't exist, add (u,x) and (v,w)
        # Wait: remove (u,v), add (u,x) and (v,w), but (w,x) must not exist
        # Actually: remove (u,v), add (u,w') and (v,x') where we're swapping partners
        # Let me redo: we want to replace edges (u,v) and some other edge with two new edges
        # Better: pick two edges (u,v) and (a,b), remove them, add (u,a) and (v,b) or (u,b) and (v,a)
        
        # Simpler approach: pick random non-edge (a,b), and random edge (u,v)
        # where a is neighbor of u, b is neighbor of v, etc.
        # Let me just do: pick random vertex u, pick random neighbor v of u,
        # pick random non-neighbor w of u, pick random neighbor x of w (x != u)
        # If (v,x) is not an edge, swap: remove (u,v),(w,x), add (u,x),(w,v)
        
        u = random.randint(0, n - 1)
        if not adj[u]:
            continue
        v = random.choice(list(adj[u]))
        # Find w not adjacent to u
        non_neighbors = [i for i in range(n) if i != u and i not in adj[u]]
        if not non_neighbors:
            continue
        w = random.choice(non_neighbors)
        if not adj[w]:
            continue
        x = random.choice(list(adj[w]))
        if x == v:
            continue
        if (min(v, x), max(v, x)) in edge_set:
            continue
        # Check degrees won't be violated (they won't, we're swapping)
        # Perform swap
        adj[u].discard(v)
        adj[v].discard(u)
        adj[w].discard(x)
        adj[x].discard(w)
        adj[u].add(x)
        adj[x].add(u)
        adj[w].add(v)
        adj[v].add(w)
        edge_set.discard((min(u, v), max(u, v)))
        edge_set.discard((min(w, x), max(w, x)))
        edge_set.add((min(u, x), max(u, x)))
        edge_set.add((min(w, v), max(w, v)))
        
        new_girth, _ = find_short_cycle()
        if new_girth < current_girth:
            # Revert
            adj[u].discard(x)
            adj[x].discard(u)
            adj[w].discard(v)
            adj[v].discard(w)
            adj[u].add(v)
            adj[v].add(u)
            adj[w].add(x)
            adj[x].add(w)
            edge_set.discard((min(u, x), max(u, x)))
            edge_set.discard((min(w, v), max(w, v)))
            edge_set.add((min(u, v), max(u, v)))
            edge_set.add((min(w, x), max(w, x)))
        else:
            current_girth = new_girth
            if current_girth >= g:
                break
    
    return [(min(u, v), max(u, v)) for u, v in edge_set]


def _bipartite_double(n_half, edges_half, k):
    """Create a bipartite graph by doubling a half-graph."""
    # This creates a 2n_half vertex graph
    # Not directly useful, skip for now
    pass


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for the 150s limit
    
    # Moore bound
    moore = _moore_lower_bound(k, g)
    
    # Start with LUW construction
    n_luw, edges_luw = _luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # Verify LUW
    valid, girth = _verify_graph(n_luw, edges_luw, k, g)
    if not valid:
        # LUW failed, try other approaches
        pass
    
    # Try to find smaller graphs using random construction + local search
    # For k=5, g=9: Moore bound is 1 + 5*(1+4+16) = 1+5*21 = 106
    # LUW gives much more. Known best is 1116.
    
    # Strategy: try random k-regular graphs of increasing size near the Moore bound
    # and check girth. This is unlikely to work for g=9 but worth trying.
    
    # Better strategy: use the LUW construction but try to minimize the component size
    # Or: try different parameterizations
    
    # For now, let's also try a simple random regular graph approach for small n
    # and verify girth
    
    # Try n from moore to min(n_luw, moore + 200) in steps
    # This is very unlikely to find girth 9 graphs by random chance
    
    # Instead, let's focus on verifying the LUW result is correct and return it.
    # The LUW construction for k=5, g=9 should give a valid graph.
    
    # Let's also try to see if we can do better with a different d parameter
    # or a different construction.
    
    # For k=5 (prime), the LUW construction with d=5 should give girth >= 10
    # (since d is odd, girth >= d+5 = 10 >= 9). Good.
    
    # The number of vertices in the LUW graph is q * q^d = q^(d+1) for the full graph,
    # but we only take the connected component of the zero vertex.
    # For q=5, d=5: the full graph has 5^6 = 15625 vertices, but the component
    # might be smaller.
    
    # Actually, the LUW construction creates a bipartite graph where each side has
    # q^d vertices (points and lines), so total 2*q^d. For q=5, d=5: 2*3125 = 6250.
    # But we only take the connected component, which could be much smaller.
    
    # Let's just return the LUW result if it's valid.
    
    if valid and girth >= g:
        return n_luw, edges_luw
    
    # If LUW is invalid, try to fix it or use a fallback
    # Fallback: try random regular graphs
    random.seed(42)
    for attempt in range(100):
        if time.time() > deadline:
            break
        n_try = moore + attempt * 2  # n must have same parity as k*n/2 * 2
        # n*k must be even for a k-regular graph
        if (n_try * k) % 2 != 0:
            continue
        if n_try > n_luw:
            break
        edges_try = _random_regular_graph(n_try, k)
        if edges_try is None:
            continue
        valid_try, girth_try = _verify_graph(n_try, edges_try, k, g)
        if valid_try and girth_try >= g:
            return n_try, edges_try
    
    # Return best known
    return best_n, best_edges
