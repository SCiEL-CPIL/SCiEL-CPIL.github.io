"""Deterministic construction for k-regular graphs with high girth.
Combines LUW algebraic construction, circulant search, and lift-based search.
"""
import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    def check_deadline():
        return time.time() > deadline

    # For the specific case k=3, g=16:
    if k == 3 and g == 16:
        # Strategy 1: LUW construction with various d values
        for d in [11, 13, 9, 15, 7]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and n_luw > 0:
                    if _verify_girth(n_luw, edges_luw, 16):
                        return n_luw, edges_luw
            except Exception:
                pass

        # Strategy 2: Circulant graph search C_n(1, a, b)
        try:
            n_circ, edges_circ = _find_circulant_cage(3, 16, deadline)
            if n_circ is not None:
                return n_circ, edges_circ
        except Exception:
            pass

        # Strategy 3: Lift-based search (Petersen base)
        try:
            n_lift, edges_lift = _lift_search(3, 16, deadline)
            if n_lift is not None:
                return n_lift, edges_lift
        except Exception:
            pass

    # General fallback: LUW
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # make it odd
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
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

    # Build the rule for each column c in [2, d]
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Total vertices: 2 * q^d
    # Vertex encoding: (side, x_1, x_2, ..., x_d) where side in {0,1}
    # We use integer encoding: vertex = side * q^d + sum(x_i * q^(i-1))
    qd = q ** d
    n = 2 * qd

    # Precompute powers of q
    qpow = [1] * (d + 1)
    for i in range(1, d + 1):
        qpow[i] = qpow[i-1] * q

    def decode(vertex):
        side = vertex // qd
        x = vertex % qd
        coords = [0] * (d + 1)  # 1-indexed
        for i in range(d, 0, -1):
            coords[i] = x % q
            x //= q
        return side, coords

    def encode(side, coords):
        x = 0
        for i in range(1, d + 1):
            x += coords[i] * qpow[i-1]
        return side * qd + x

    edges = []
    seen_edges = set()

    for side in range(2):
        for x_base in range(qd):
            # Decode x_base into coords
            coords = [0] * (d + 1)
            tmp = x_base
            for i in range(d, 0, -1):
                coords[i] = tmp % q
                tmp //= q

            v1 = encode(side, coords)

            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(coords[c], mul(y[a], coords[b]))
                    else:
                        y[c] = sub(coords[c], mul(coords[a], y[b]))

                v2 = encode(1 - side, y)

                if v1 < v2:
                    if (v1, v2) not in seen_edges:
                        seen_edges.add((v1, v2))
                        edges.append((v1, v2))
                elif v2 < v1:
                    if (v2, v1) not in seen_edges:
                        seen_edges.add((v2, v1))
                        edges.append((v2, v1))

    return n, edges


def _find_circulant_cage(k, g, deadline):
    """
    Search for a circulant graph C_n(1, a, b) with girth at least g.
    Assumes k=3.
    """
    # Moore bound for k=3, g=16 (even girth):
    # n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * (2^8 - 1) = 510
    # n must be even. So start from 512.
    min_n = 512
    max_n = 700

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 10:
            return None, None

        # Generate candidate (a, b) pairs
        # Heuristic: a and b should be "far apart" to avoid short cycles
        candidates = []
        for a in range(2, n // 2):
            for b in range(a + 1, n // 2):
                # Quick filters
                if _quick_circulant_filter(n, a, b, g):
                    continue
                candidates.append((a, b))

        # Sort by a heuristic score (prefer larger gaps)
        candidates.sort(key=lambda ab: -(ab[0] + ab[1]))

        # Try top candidates
        for a, b in candidates[:50]:
            if time.time() > deadline - 10:
                return None, None

            if _has_short_cycle_circulant(n, a, b, g):
                continue

            # Construct edges
            edges = []
            seen = set()
            for i in range(n):
                for shift in [1, a, b]:
                    j = (i + shift) % n
                    if i < j:
                        if (i, j) not in seen:
                            seen.add((i, j))
                            edges.append((i, j))
                    elif j < i:
                        if (j, i) not in seen:
                            seen.add((j, i))
                            edges.append((j, i))

            return n, edges

    return None, None


def _quick_circulant_filter(n, a, b, g):
    """Quick check to filter out obviously bad (a, b) pairs."""
    gens = {1, n-1, a, n-a, b, n-b}
    # Check for 4-cycles: sum of two generators equals another generator or 0
    for g1 in gens:
        for g2 in gens:
            s = (g1 + g2) % n
            if s == 0:
                continue
            if s in gens:
                return True
    return False


def _has_short_cycle_circulant(n, a, b, g):
    """
    Check if the circulant graph C_n(1, a, b) has a cycle shorter than g.
    Uses BFS from 0.
    Returns True if a short cycle is found, False otherwise.
    """
    visited = [-1] * n
    parent = [-1] * n
    visited[0] = 0
    queue = deque([(0, 0)])

    max_depth = g // 2

    while queue:
        u, depth = queue.popleft()

        for shift in [1, a, b]:
            v1 = (u + shift) % n
            v2 = (u - shift) % n

            for v in [v1, v2]:
                if visited[v] == -1:
                    visited[v] = depth + 1
                    parent[v] = u
                    if depth + 1 <= max_depth:
                        queue.append((v, depth + 1))
                else:
                    if parent[v] == u or parent[u] == v:
                        continue
                    cycle_len = depth + 1 + visited[v]
                    if cycle_len < g:
                        return True

    return False


def _lift_search(k, g, deadline):
    """Lift-based search using Petersen graph as base."""
    # Petersen graph edges (0-9)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))

    m = 52
    while m <= 1500 and time.time() < deadline - 5:
        # Deterministic 1: Linear spread
        shifts1 = [int((i + 1) * m / 15.0) for i in range(15)]
        n, edges = _build_lift(base_edges, m, shifts1)
        if _verify_girth(n, edges, g):
            return n, edges

        # Deterministic 2: Golden ratio based shifts
        golden = (1 + 5**0.5) / 2.0
        shifts2 = [int((i + 1) * m * golden % m) for i in range(15)]
        n, edges = _build_lift(base_edges, m, shifts2)
        if _verify_girth(n, edges, g):
            return n, edges

        # Random shifts
        num_random_tries = 20
        for _ in range(num_random_tries):
            if time.time() > deadline - 5:
                break
            shifts_rand = [random.randint(0, m-1) for _ in range(15)]
            n, edges = _build_lift(base_edges, m, shifts_rand)
            if _verify_girth(n, edges, g):
                return n, edges

        m += 1

    return None, None


def _build_lift(base_edges, m, shifts):
    """Build the lifted graph."""
    n = 10 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges


def _verify_girth(n, edges, g):
    """
    Fast girth verification using timestamped BFS with early termination.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dist = [0] * n
    parent = [-1] * n
    seen_mark = [0] * n
    mark = 0

    for s in range(n):
        mark += 1
        dist[s] = 0
        parent[s] = -1
        seen_mark[s] = mark
        queue = deque([s])

        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g // 2:
                continue
            for v in adj[u]:
                if seen_mark[v] != mark:
                    dist[v] = du + 1
                    parent[v] = u
                    seen_mark[v] = mark
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False

    return True

# EVOLVE-BLOCK-END
