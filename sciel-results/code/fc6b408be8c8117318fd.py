"""Combined program: algebraic voltage lifts + fast verification + LUW fallback.
Structural change: use a 3-regular base with girth 8 (Heawald) instead of Petersen (girth 5),
so the base graph itself contributes more to the girth before lift cycles can close.
Also: avoid zero shifts (which create short cycles), use more random attempts,
and add a Heawald-graph base with carefully chosen voltage patterns.
"""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _quadratic_residues(m, count):
    """Generate count distinct quadratic residues modulo m, spread out, avoiding 0."""
    if m <= 1:
        return [0] * count
    residues = set()
    for a in range(1, m):
        residues.add((a * a) % m)
    for a in range(1, m):
        residues.add((m - (a * a) % m) % m)
    # Remove 0 to avoid short cycles
    residues.discard(0)
    residues_list = sorted(residues)
    if len(residues_list) < count:
        residues_list = [x for x in range(1, m)]
    step = max(1, len(residues_list) // count)
    result = []
    for i in range(count):
        idx = (i * step) % len(residues_list)
        result.append(residues_list[idx])
    return result

def _verify_girth_fast(adj, n, g):
    """Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _build_lift(base_edges, m, shifts):
    """Build a voltage graph lift over Z_m.
    base_edges: list of (u,v) on the base graph
    m: lift size
    shifts: list of voltage shifts, one per base edge
    Returns (n, adj_list, edge_list) or None if invalid."""
    if not base_edges:
        return None
    n_base = max(max(u for u, v in base_edges), max(v for u, v in base_edges)) + 1
    n = n_base * m
    adj = [set() for _ in range(n)]
    edges = []
    for idx, (u, v) in enumerate(base_edges):
        vol = shifts[idx] % m
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    adj_list = [list(s) for s in adj]
    return n, adj_list, edges

def _luw_construct(k, g, d_override=None):
    """LUW construction with optional d override."""
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw

    # k=3, target girth g (typically 16)
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2

    # Base graphs to try (all 3-regular)
    # Heawald graph: 14 vertices, 21 edges, girth 6
    # It's the incidence graph of Fano plane
    heawald_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
        (0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12), (6, 13)
    ]
    # Actually let me use the standard Heawald: it's the bipartite graph of Fano plane
    # 7 points, 7 lines, each point on 3 lines, each line has 3 points
    # Vertices: 0-6 are points, 7-13 are lines
    # Fano plane lines: {0,1,2}, {0,3,4}, {0,5,6}, {1,3,5}, {1,4,6}, {2,3,6}, {2,4,5}
    heawald_edges = [
        # Point 0 on lines 0,1,2
        (0, 7), (0, 8), (0, 9),
        # Point 1 on lines 0,3,4
        (1, 7), (1, 10), (1, 11),
        # Point 2 on lines 0,5,6
        (2, 7), (2, 12), (2, 13),
        # Point 3 on lines 1,3,5
        (3, 8), (3, 10), (3, 12),
        # Point 4 on lines 1,4,6
        (4, 8), (4, 11), (4, 13),
        # Point 5 on lines 2,3,6
        (5, 9), (5, 10), (5, 13),
        # Point 6 on lines 2,4,5
        (6, 9), (6, 11), (6, 12)
    ]

    # Petersen graph (10 vertices, 15 edges, girth 5)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    # K_3,3 (6 vertices, 9 edges, girth 4)
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]

    # Triangular prism: 6 vertices, 9 edges, girth 4
    prism_edges = [
        (0,1),(1,2),(2,0),  # top triangle
        (3,4),(4,5),(5,3),  # bottom triangle
        (0,3),(1,4),(2,5)   # verticals
    ]

    # Cube graph Q3: 8 vertices, 12 edges, girth 4
    cube_edges = [
        (0,1),(0,2),(0,4),
        (1,3),(1,5),
        (2,3),(2,6),
        (3,7),
        (4,5),(4,6),
        (5,7),
        (6,7)
    ]

    base_graphs = [
        ("heawald", heawald_edges, 14),
        ("petersen", petersen_edges, 10),
        ("prism", prism_edges, 6),
        ("cube", cube_edges, 8),
        ("k33", k33_edges, 6),
    ]

    best_result = None
    best_n = float('inf')

    for name, base_edges, n_base in base_graphs:
        if _time.time() > deadline - 10:
            break
        e_count = len(base_edges)
        n_min = max(1, (moore + n_base - 1) // n_base)

        # Try m values from n_min upward, with a wider range
        for m in range(n_min, n_min + 500):
            if _time.time() > deadline - 5:
                break
            n = n_base * m
            if n >= best_n:
                continue

            # Deterministic shifts: quadratic residues (avoiding 0)
            shifts_det = _quadratic_residues(m, e_count)
            result = _build_lift(base_edges, m, shifts_det)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    best_n = n_r
                    best_result = (n_r, edges)
                    return n_r, edges

            # Second deterministic: linear spread (avoiding 0)
            shifts_lin = [int((i + 1) * m / (e_count + 1)) % m for i in range(e_count)]
            # Ensure no zeros
            shifts_lin = [s if s != 0 else 1 for s in shifts_lin]
            result = _build_lift(base_edges, m, shifts_lin)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    best_n = n_r
                    best_result = (n_r, edges)
                    return n_r, edges

            # Random shifts: multiple attempts, avoid zero
            max_attempts = 20
            for attempt in range(max_attempts):
                if _time.time() > deadline - 3:
                    break
                rng = _random.Random(hash((name, m, attempt)) & 0xFFFFFFFF)
                shifts_rand = [rng.randint(1, m - 1) for _ in range(e_count)]
                result = _build_lift(base_edges, m, shifts_rand)
                if result is not None:
                    n_r, adj, edges = result
                    if _verify_girth_fast(adj, n_r, g):
                        best_n = n_r
                        best_result = (n_r, edges)
                        return n_r, edges

    # Fallback: LUW with various d values
    if _time.time() < deadline - 5:
        for d_try in range(6, 12):
            if _time.time() > deadline - 3:
                break
            try:
                n_try, edges_try = _luw_construct(3, g, d_override=d_try)
                if n_try < best_n:
                    adj = [set() for _ in range(n_try)]
                    for u, v in edges_try:
                        adj[u].add(v)
                        adj[v].add(u)
                    adj_list = [list(s) for s in adj]
                    if _verify_girth_fast(adj_list, n_try, g):
                        best_n = n_try
                        best_result = (n_try, edges_try)
                        return n_try, edges_try
            except Exception:
                pass

    # Final fallback: standard LUW
    if best_result is not None:
        return best_result

    n_luw, edges_luw = _luw_construct(3, g)
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
