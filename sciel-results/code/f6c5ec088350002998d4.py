"""Combined program: algebraic voltage lifts + fast verification + LUW fallback."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _quadratic_residues(m, count):
    """Generate count distinct quadratic residues modulo m, spread out."""
    if m <= 1:
        return [0] * count
    residues = set()
    for a in range(1, m):
        residues.add((a * a) % m)
    # Also include 0 and negatives for variety
    for a in range(1, m):
        residues.add((m - (a * a) % m) % m)
    residues_list = sorted(residues)
    if len(residues_list) < count:
        # Pad with linear spread if not enough distinct residues
        residues_list = list(range(m))
    # Take count distinct values, evenly spaced from the available set
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
    n_base = max(max(u for u,v in base_edges), max(v for u,v in base_edges)) + 1
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
    # Convert to lists for faster verification
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
        # LUW fallback for other k
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw

    # k=3, target girth g (typically 16)
    # Moore bound for k=3, even g: 2*(2^(g/2)-1)
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2

    # Base graphs to try (all 3-regular, small, with known structure)
    # Petersen graph (10 vertices, 15 edges)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    # K_3,3 (6 vertices, 9 edges)
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    # K_4 (4 vertices, 6 edges) - note: girth 3, lifts may have issues
    k4_edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

    base_graphs = [
        ("petersen", petersen_edges, 10),
        ("k33", k33_edges, 6),
        ("k4", k4_edges, 4),
    ]

    best_result = None
    best_n = float('inf')

    # For each base graph, try lift sizes m
    for name, base_edges, n_base in base_graphs:
        if _time.time() > deadline - 10:
            break
        e_count = len(base_edges)
        n_min = max(1, (moore + n_base - 1) // n_base)

        # Try m values from n_min upward
        # Use deterministic + random shifts
        for m in range(n_min, n_min + 300):
            if _time.time() > deadline - 5:
                break
            n = n_base * m
            if n >= best_n:
                continue

            # Deterministic shifts: quadratic residues for structural anti-short-cycle property
            shifts_det = _quadratic_residues(m, e_count)
            result = _build_lift(base_edges, m, shifts_det)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    best_n = n_r
                    best_result = (n_r, edges)
                    return n_r, edges

            # Second deterministic variant: negated quadratic residues
            shifts_det2 = [(-r) % m for r in shifts_det]
            result = _build_lift(base_edges, m, shifts_det2)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    best_n = n_r
                    best_result = (n_r, edges)
                    return n_r, edges

            # Third variant: mixed quadratic + linear
            shifts_det3 = [(shifts_det[i] + i * 7) % m for i in range(e_count)]
            result = _build_lift(base_edges, m, shifts_det3)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    best_n = n_r
                    best_result = (n_r, edges)
                    return n_r, edges

            # Random shifts: multiple attempts (reduced to save time for more m values)
            max_attempts = 5
            for attempt in range(max_attempts):
                if _time.time() > deadline - 3:
                    break
                rng = _random.Random(hash((name, m, attempt)))
                shifts_rand = [rng.randint(0, m - 1) for _ in range(e_count)]
                result = _build_lift(base_edges, m, shifts_rand)
                if result is not None:
                    n_r, adj, edges = result
                    if _verify_girth_fast(adj, n_r, g):
                        best_n = n_r
                        best_result = (n_r, edges)
                        return n_r, edges

    # Fallback: LUW with various d values (start from smaller d for fewer vertices)
    if _time.time() < deadline - 5:
        for d_try in [7, 8, 9, 10, 11, 12, 13, 14]:
            if _time.time() > deadline - 3:
                break
            try:
                n_try, edges_try = _luw_construct(3, g, d_override=d_try)
                if n_try < best_n:
                    # Verify
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
