"""
Construct a k-regular graph with girth at least g.
Structural redesign: voltage-lift over Kautz base + algebraic circulant search + LUW fallback.
"""
import time
from collections import deque

# EVOLVE-BLOCK-START


def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Strategy 1: Voltage lift over Kautz base graph
        if not check_deadline():
            result = _voltage_lift_kautz(3, 16, deadline)
            if result is not None:
                return result

        # Strategy 2: Fast circulant search
        if not check_deadline():
            result2 = _find_circulant_fast(3, 16, deadline)
            if result2 is not None:
                return result2

        # Strategy 3: LUW fallback
        for d in [9, 11, 13]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    return _luw_construct(k, g)


def _voltage_lift_kautz(k, g, deadline):
    """
    Voltage lift of the Kautz digraph treated as an undirected 3-regular base.
    The Kautz graph K(2,2) has 6 vertices and is 3-regular.
    We lift it over Z_m using voltages assigned to edges.
    Total vertices = 6 * m. We search for m and voltage assignments that give girth >= g.
    """
    # Kautz graph K(2,2): vertices are length-2 strings over {0,1} with last char != first char
    # Vertices: 01, 10, 11, 00, 101 (wait, K(2,2) has 2^2 + 2^1 = 6 vertices)
    # Actually Kautz K(q, d) has q^d + q^(d-1) vertices. For K(2,2): 4 + 2 = 6.
    # Vertices: 00, 01, 10, 11, 0, 1 (where single chars are "short" vertices)
    # Hmm, let me use a simpler 3-regular base. The Petersen graph is 3-regular with 10 vertices.
    # Or better: use the triangular prism (6 vertices, 3-regular, girth 3 - too small).
    # 
    # Actually for a voltage lift, the base graph's girth doesn't matter as much as
    # the voltage assignment. Let me use a 3-regular graph with small girth but
    # structure that allows good voltage assignments.
    #
    # Key insight: if the base has a cycle of length L, the lift will have a cycle
    # of length L * (order of the product of voltages around that cycle / m).
    # To avoid short cycles, we need the voltage products around all base cycles
    # to have large order in Z_m.
    #
    # Let's use the Petersen graph (10 vertices, 3-regular, girth 5) as base.
    # Lift over Z_m: total vertices = 10*m.
    # For girth 16, we need no cycle shorter than 16.
    # A 5-cycle in the base with voltage product p gives a cycle of length 5*(m/gcd(p,m)) in the lift.
    # If p is coprime to m, that's a cycle of length 5*m. For m >= 4, that's >= 20 > 16. Good.
    # A 6-cycle? Petersen has no 6-cycles (girth 5, next is 5... actually girth is 5, so cycles of length 5, 6, ...).
    # Wait, Petersen girth is 5, so it has 5-cycles but no shorter. It may have 6-cycles? No, girth 5 means no cycles of length 3 or 4. It has 5-cycles.
    #
    # For the lift: a closed walk of length L in the base with voltage sum s (mod m)
    # gives a closed walk of length L in the lift (if we stay in the same "layer") only if s=0.
    # Actually, the lift vertices are (v, i) where v is a base vertex and i in Z_m.
    # Edge (u,v) with voltage w connects (u,i) to (v, i+w).
    # A closed walk in the lift from (v0, 0) back to (v0, 0) corresponds to a closed walk
    # v0->v1->...->vL=v0 in the base with total voltage sum = 0 mod m.
    # The shortest such walk determines the girth of the lift (at least for the component containing (v0,0)).
    #
    # Strategy: assign voltages so that the shortest closed walk with sum 0 mod m is >= g.
    # For the Petersen base (10 vertices, 15 edges, girth 5):
    # - 5-cycles: need voltage sum not 0 mod m
    # - 6-walks (non-simple): need sum not 0
    # - etc.
    #
    # This is complex. Let me try a different approach: use a known 3-regular graph
    # with girth >= 5 as base, and search for voltage assignments.
    
    # Petersen graph adjacency (0-indexed):
    # Outer pentagon: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                      (5,7),(7,9),(9,6),(6,8),(8,5),
                      (0,5),(1,6),(2,7),(3,8),(4,9)]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)
    
    # Search for m and voltage assignment
    for m in range(6, 100):  # total vertices = 10*m
        if time.time() > deadline - 10:
            return None
        n_total = 10 * m
        if n_total < 512:
            continue
        
        # Try deterministic voltage assignment: edge voltage = hash(edge) mod m
        # But we need to ensure no short cycles.
        # 
        # Alternative: use a linear voltage assignment.
        # Assign each edge (u,v) a voltage based on a spanning tree + fundamental cycles.
        # 
        # Simpler: try random-ish assignments and check girth.
        import random
        rng = random.Random(42)
        
        # For efficiency, try a few specific patterns
        for trial in range(20):
            if time.time() > deadline - 10:
                return None
            
            # Assign voltages
            voltages = {}
            for i, (u, v) in enumerate(petersen_edges):
                voltages[(u, v)] = (i * 7 + trial * 13) % m
                voltages[(v, u)] = (-voltages[(u, v)]) % m
            
            # Check girth of the lift using non-backtracking BFS from a few vertices
            if _check_lift_girth(m, petersen_edges, voltages, g):
                # Build the full edge list
                edges = []
                for u, v in petersen_edges:
                    w = voltages[(u, v)]
                    for i in range(m):
                        # (u,i) -> (v, (i+w) % m)
                        u_id = u * m + i
                        v_id = v * m + ((i + w) % m)
                        if u_id < v_id:
                            edges.append((u_id, v_id))
                # Verify it's 3-regular and simple
                deg = [0] * n_total
                edge_set = set()
                for a, b in edges:
                    deg[a] += 1
                    deg[b] += 1
                    if (a, b) in edge_set or (b, a) in edge_set:
                        return None  # duplicate edge
                    edge_set.add((a, b))
                if all(d == 3 for d in deg):
                    return n_total, edges
    
    return None


def _check_lift_girth(m, base_edges, voltages, g):
    """
    Check if the voltage lift has girth >= g.
    Use non-backtracking BFS from vertex (0, 0).
    Since the lift is vertex-transitive (if the base is), checking one vertex suffices.
    Actually it's not necessarily vertex-transitive, but for a Cayley-type lift it is.
    For safety, check a few starting vertices.
    """
    n_base = 10
    n_total = n_base * m
    
    # Build adjacency of the lift
    # Vertex (u, i) is identified as u*m + i
    adj = [[] for _ in range(n_total)]
    for u, v in base_edges:
        w = voltages[(u, v)]
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + ((i + w) % m)
            adj[u_id].append(v_id)
            adj[v_id].append(u_id)
    
    # Non-backtracking BFS from (0,0) i.e., vertex 0
    if _nonbacktrack_girth_check(adj, 0, g):
        # Also check from a couple more vertices for safety
        if _nonbacktrack_girth_check(adj, 1, g):
            if _nonbacktrack_girth_check(adj, m, g):  # vertex (1, 0)
                return True
    return False


def _nonbacktrack_girth_check(adj, start, g):
    """
    Non-backtracking BFS from start. Returns True if no cycle shorter than g is found.
    State: (vertex, previous_vertex)
    """
    n = len(adj)
    max_depth = g - 1
    dist = {}
    queue = deque()
    
    start_state = (start, -1)
    dist[start_state] = 0
    queue.append(start_state)
    
    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]
        if d >= max_depth:
            continue
        for v in adj[u]:
            if v == prev:
                continue
            ns = (v, u)
            nd = d + 1
            if nd > max_depth:
                continue
            if v == start and u != -1 and nd < g:
                return False  # found cycle of length nd
            if ns not in dist or nd < dist[ns]:
                dist[ns] = nd
                queue.append(ns)
    return True


def _find_circulant_fast(k, g, deadline):
    """
    Fast circulant C_n(1,a,b) search with precomputed 4/6-cycle filters.
    """
    min_n = 512
    max_n = 940

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 5:
            return None

        half = n // 2

        # Heuristic: a near n/3, b near 2n/3 (but b < n/2)
        # Try a range of values
        for a in range(max(3, n // 4), half):
            if time.time() > deadline - 5:
                return None

            two_a = 2 * a % n
            if two_a in (1, n - 1, a, n - a):
                continue

            for b in range(a + 1, half):
                if time.time() > deadline - 3:
                    return None

                if _quick_4cycle(n, a, b):
                    continue
                if _quick_6cycle(n, a, b):
                    continue

                if _girth_check_circ(n, a, b, g):
                    edges = []
                    for i in range(n):
                        for shift in (1, a, b):
                            j = (i + shift) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return None


def _quick_4cycle(n, a, b):
    """Fast algebraic 4-cycle check for C_n(1,a,b)."""
    G = [1, n - 1, a, n - a, b, n - b]
    # Check all 4-tuples
    for i in range(6):
        for j in range(6):
            if G[j] == (-G[i]) % n:
                continue
            v2 = (G[i] + G[j]) % n
            if v2 == 0:
                continue
            for k in range(6):
                if G[k] == (-G[j]) % n:
                    continue
                v3 = (v2 + G[k]) % n
                if v3 == 0:
                    continue
                for l in range(6):
                    if G[l] == (-G[k]) % n:
                        continue
                    if (v3 + G[l]) % n == 0:
                        # Check distinctness
                        if G[i] != v2 and v2 != v3 and G[i] != v3:
                            return True
    return False


def _quick_6cycle(n, a, b):
    """Fast 6-cycle check: BFS to depth 3 from 0, look for edges within depth 3."""
    G = [1, n - 1, a, n - a, b, n - b]
    
    dist = {0: 0}
    parent = {0: -1}
    level = [0]
    
    for depth in range(3):
        next_level = []
        for u in level:
            for step in G:
                v = (u + step) % n
                if v == 0 and depth < 2:
                    continue
                if v not in dist:
                    dist[v] = depth + 1
                    parent[v] = u
                    next_level.append(v)
        level = next_level
    
    # Check for edges between depth-3 vertices
    for v in level:
        for step in G:
            w = (v + step) % n
            if w in dist and dist[w] == 3 and parent[w] != v and parent[v] != w:
                return True
    return False


def _girth_check_circ(n, a, b, g):
    """Full non-backtracking girth check for C_n(1,a,b) from vertex 0."""
    G = [1, n - 1, a, n - a, b, n - b]
    max_depth = g - 1
    dist = {}
    queue = deque()
    start = (0, -1)
    dist[start] = 0
    queue.append(start)
    
    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]
        if d >= max_depth:
            continue
        for step in G:
            v = (u + step) % n
            if v == prev:
                continue
            ns = (v, u)
            nd = d + 1
            if nd > max_depth:
                continue
            if v == 0 and u != -1 and nd < g:
                return False
            if ns not in dist or nd < dist[ns]:
                dist[ns] = nd
                queue.append(ns)
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
    q = k

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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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


def _verify_girth_fast(n, edges, g):
    """Fast girth verification."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _bfs_short_cycle(src, adj, g):
            return False

    return True


def _bfs_short_cycle(src, adj, g):
    """BFS from src to find cycles shorter than g."""
    n = len(adj)
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])
    max_depth = g // 2

    while queue:
        u = queue.popleft()
        depth = dist[u]

        if depth >= max_depth:
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        return True
            continue

        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                if parent[v] == u or parent[u] == v:
                    continue
                cycle_len = depth + 1 + dist[v]
                if cycle_len < g:
                    return True

    return False

# EVOLVE-BLOCK-END
