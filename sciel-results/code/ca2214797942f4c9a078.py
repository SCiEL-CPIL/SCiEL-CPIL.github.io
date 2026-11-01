"""Construct small 3-regular graphs of large girth via random permutation groups and voltage lifts."""
import time
import random
import numpy as np
from collections import deque

def _verify_girth(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
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

def _verify_girth_fast(adj_arr, n, g):
    """Fast girth verification using numpy arrays. adj_arr is (n, 3) array."""
    for start in range(n):
        # State: (vertex, prev_vertex) -> distance
        # Use dict for sparse representation
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj_arr[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _check_girth_perm(gens, n, g):
    """Check girth >= g for a graph defined by 3 generators (permutations).
    gens: list of 3 permutations, each a numpy array of length n where gens[i][v] = neighbor.
    Returns True if girth >= g."""
    # The graph has edges (v, gens[i][v]) for i in 0,1,2
    # Check for short cycles using BFS with non-backtracking
    for start in range(n):
        # State: (current_vertex, prev_vertex, gen_used_to_arrive)
        # Actually simpler: (current, prev) is enough since graph is 3-regular
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for i in range(3):
                w = gens[i][v]
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _make_perm_from_gens(gens, n):
    """Verify generators form a valid 3-regular graph.
    Each gen is an involution (self-inverse) with no fixed points.
    No two generators are the same.
    Returns (adj_list, edges) if valid, else None."""
    # Check involutions
    for i, gen in enumerate(gens):
        if any(gen[v] == v for v in range(n)):
            return None
        for v in range(n):
            if gen[gen[v]] != v:
                return None
    # Check distinct
    if len(set(tuple(g) for g in gens)) != 3:
        return None
    # Build adjacency
    adj = [set() for _ in range(n)]
    for gen in gens:
        for v in range(n):
            adj[v].add(gen[v])
    for v in range(n):
        if len(adj[v]) != 3:
            return None
    adj_list = [list(s) for s in adj]
    edges = []
    for v in range(n):
        for w in adj_list[v]:
            if v < w:
                edges.append((v, w))
    return adj_list, edges

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
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw

    # k=3, target girth g (typically 16)
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2

    best_result = None
    best_n = float('inf')

    # === STRATEGY 1: Random permutation group search ===
    # Generate random 3-generator groups on n vertices, check girth
    # n must be even (for 3-regular graph to exist)
    # Try n values from Moore bound upward
    # Use multiple of 6 for better symmetry properties
    
    # First try: direct random permutation search
    # For each candidate n, try several random generator sets
    n_candidates = []
    # Start from Moore bound, try multiples of 6 and nearby
    for n_try in range(max(moore, 6), max(moore, 6) + 300):
        if n_try % 2 == 0:  # must be even for 3-regular
            n_candidates.append(n_try)
    
    # Also try specific values that are multiples of small numbers
    for mult in [6, 10, 12, 14, 15, 18, 20, 21, 24, 28, 30]:
        for base in range(max(moore // mult, 1), max(moore // mult, 1) + 10):
            n_try = base * mult
            if n_try >= max(moore, 6) and n_try < max(moore, 6) + 400 and n_try % 2 == 0:
                if n_try not in n_candidates:
                    n_candidates.append(n_try)
    
    n_candidates = sorted(set(n_candidates))
    
    # For each candidate n, try random permutations
    max_per_n = 50  # max random attempts per n value
    for n_try in n_candidates:
        if time.time() > deadline - 15:
            break
        if n_try >= best_n:
            continue
        
        # Generate random 3-regular graph via random perfect matchings
        # A 3-regular graph can be decomposed into 3 perfect matchings
        # Each perfect matching is an involution with no fixed points
        attempts = 0
        while attempts < max_per_n and time.time() < deadline - 15:
            attempts += 1
            rng = random.Random(hash(('perm', n_try, attempts)))
            
            # Generate 3 random perfect matchings
            gens = []
            valid = True
            for i in range(3):
                # Generate random perfect matching
                perm = list(range(n_try))
                rng.shuffle(perm)
                # Pair up: 0-1, 2-3, etc.
                gen = [0] * n_try
                for j in range(0, n_try, 2):
                    a, b = perm[j], perm[j+1]
                    gen[a] = b
                    gen[b] = a
                gens.append(np.array(gen))
            
            # Check validity: involutions, no fixed points, distinct
            valid = True
            for gen in gens:
                if any(gen[v] == v for v in range(n_try)):
                    valid = False
                    break
            if not valid:
                continue
            # Check distinct
            if len(set(tuple(g.tolist()) for g in gens)) != 3:
                continue
            # Check no overlapping edges
            for i in range(3):
                for j in range(i+1, 3):
                    for v in range(n_try):
                        if gens[i][v] == gens[j][v]:
                            valid = False
                            break
                        if gens[i][v] == gens[j][gens[j][v]]:
                            # This is fine, just means they share a vertex
                            pass
                    if not valid:
                        break
            if not valid:
                continue
            
            # Check girth
            if _check_girth_perm(gens, n_try, g):
                # Build edge list
                adj = [set() for _ in range(n_try)]
                for gen in gens:
                    for v in range(n_try):
                        adj[v].add(int(gen[v]))
                edges = []
                for v in range(n_try):
                    for w in adj[v]:
                        if v < w:
                            edges.append((v, w))
                best_n = n_try
                best_result = (n_try, edges)
                return n_try, edges
    
    # === STRATEGY 2: Voltage lifts (original approach) ===
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    dodeca_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (5, 10), (6, 11), (7, 12), (8, 13), (9, 14),
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 10),
        (10, 15), (11, 16), (12, 17), (13, 18), (14, 19),
        (15, 16), (16, 17), (17, 18), (18, 19), (19, 15),
    ]

    base_graphs = [
        ("k33", k33_edges, 6),
        ("petersen", petersen_edges, 10),
        ("dodeca", dodeca_edges, 20),
    ]

    for name, base_edges, n_base in base_graphs:
        if time.time() > deadline - 10:
            break
        e_count = len(base_edges)
        n_min = max(1, (moore + n_base - 1) // n_base)

        for m in range(n_min, n_min + 300):
            if time.time() > deadline - 5:
                break
            n = n_base * m
            if n >= best_n:
                continue

            max_attempts = 5
            for attempt in range(max_attempts):
                if time.time() > deadline - 3:
                    break
                rng = random.Random(hash((name, m, attempt)))
                shifts_rand = [rng.randint(0, m - 1) for _ in range(e_count)]
                result = _build_lift(base_edges, m, shifts_rand)
                if result is not None:
                    n_r, adj, edges = result
                    if _verify_girth(adj, n_r, g):
                        best_n = n_r
                        best_result = (n_r, edges)
                        return n_r, edges

    # === STRATEGY 3: LUW fallback ===
    if time.time() < deadline - 5:
        for d_try in [7, 8, 9, 10, 11, 12, 13, 14]:
            if time.time() > deadline - 3:
                break
            try:
                n_try, edges_try = _luw_construct(3, g, d_override=d_try)
                if n_try < best_n:
                    adj = [set() for _ in range(n_try)]
                    for u, v in edges_try:
                        adj[u].add(v)
                        adj[v].add(u)
                    adj_list = [list(s) for s in adj]
                    if _verify_girth(adj_list, n_try, g):
                        best_n = n_try
                        best_result = (n_try, edges_try)
                        return n_try, edges_try
            except Exception:
                pass

    if best_result is not None:
        return best_result

    n_luw, edges_luw = _luw_construct(3, g)
    return n_luw, edges_luw
