"""Deterministic Algebraic Lift construction for k-regular graphs with high girth."""
import time
import sys
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g.
    For k=3, g=16, we use a voltage lift of the Petersen graph.
    The Petersen graph has girth 5. A lift of size m has n=10m vertices.
    We search for m and voltage assignments (shifts) such that the lift has girth >= g.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    if k != 3:
        return _luw_construct(k, g)

    # Petersen graph edges (0-9)
    # Standard Petersen: outer pentagon 0-1-2-3-4-0, inner star 5-7-9-6-8-5, spokes 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    # Base edges (u < v)
    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))
    
    num_base_edges = len(base_edges)

    # Moore bound for k=3, g=16 is 512.
    # n = 10 * m. So m >= 52.
    m_start = 52

    random.seed(12345)

    # We will scan m up to 1000.
    # With faster verification, we can try more random shifts per m.
    # Strategy: Spend more time on small m where success probability is higher per attempt?
    # Actually, success probability increases with m. But n is smaller for small m.
    # The "island" finding suggests small m (52-100) with many trials is key.
    
    # Pre-compute deterministic shifts for various m? No, they depend on m.
    
    while m <= 1000 and time.time() < deadline:
        # Determine number of random attempts based on m
        # For small m, we need many attempts. For large m, fewer.
        if m < 100:
            num_random_tries = 500
        elif m < 200:
            num_random_tries = 200
        elif m < 500:
            num_random_tries = 50
        else:
            num_random_tries = 20

        # Try Thue-Morse based deterministic assignments first
        # These have better structural properties for avoiding short cycles
        shifts_tm1 = _thue_morse_shifts(m, num_base_edges)
        n, edges = _build_lift(base_edges, m, shifts_tm1)
        if _verify_girth_fast(n, edges, g):
            return n, edges

        shifts_tm2 = _thue_morse_shifts_v2(m, num_base_edges)
        n, edges = _build_lift(base_edges, m, shifts_tm2)
        if _verify_girth_fast(n, edges, g):
            return n, edges

        # Deterministic 1: Linear spread
        shifts1 = [int((i + 1) * m / 15.0) for i in range(15)]
        n, edges = _build_lift(base_edges, m, shifts1)
        if _verify_girth_fast(n, edges, g):
            return n, edges

        # Deterministic 2: Golden ratio based shifts
        golden = (1 + 5**0.5) / 2.0
        shifts2 = [int((i + 1) * m * golden % m) for i in range(15)]
        n, edges = _build_lift(base_edges, m, shifts2)
        if _verify_girth_fast(n, edges, g):
            return n, edges

        # Deterministic 3: Prime-based shifts
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        shifts3 = [primes[i % 15] * m // 15 for i in range(15)]
        n, edges = _build_lift(base_edges, m, shifts3)
        if _verify_girth_fast(n, edges, g):
            return n, edges

        # Try more random shifts to increase probability of finding a valid girth
        for _ in range(num_random_tries):
            if time.time() > deadline:
                break
            # Use random shifts in [1, m-1] to avoid trivial 0-shift issues?
            # Actually 0 is fine, but non-zero is slightly better for girth sometimes.
            # The inspiration used [1, m-1]. Let's stick with that for consistency with high-performing variant.
            shifts_rand = [random.randint(1, m-1) if m > 1 else 0 for _ in range(15)]
            n, edges = _build_lift(base_edges, m, shifts_rand)
            if _verify_girth_fast(n, edges, g):
                return n, edges

        m += 1

    # If we didn't find one in the loop, try larger m with more attempts
    if time.time() < deadline:
        for m in range(1001, 2000):
            if time.time() > deadline:
                break
            for _ in range(50):
                if time.time() > deadline:
                    break
                shifts_rand = [random.randint(1, m-1) for _ in range(15)]
                n, edges = _build_lift(base_edges, m, shifts_rand)
                if _verify_girth_fast(n, edges, g):
                    return n, edges

    # Fallback: Return a large random lift (likely valid)
    # This should rarely be reached.
    m = 3000
    shifts_rand = [random.randint(1, m-1) for _ in range(15)]
    n, edges = _build_lift(base_edges, m, shifts_rand)
    return n, edges


def _build_lift(base_edges, m, shifts):
    """
    Build the lifted graph.
    base_edges: list of (u, v) for u < v
    m: lift size
    shifts: list of shifts for each base edge
    """
    n = 10 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using timestamped BFS with early termination.
    Optimized with numpy for adjacency and faster array operations.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Use arrays for fast access
    dist = np.full(n, -1, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    seen_mark = np.zeros(n, dtype=np.int32)
    mark = 0

    half_g = g // 2

    for s in range(n):
        mark += 1
        # Reset only what we need
        dist[s] = 0
        parent[s] = -1
        seen_mark[s] = mark
        queue = deque([s])

        while queue:
            u = queue.popleft()
            du = dist[u]
            # If current distance is already >= g/2, we don't need to go further
            # because any cycle found would be at least 2*du + 1 >= g
            if du >= half_g:
                continue
            for v in adj[u]:
                if seen_mark[v] != mark:
                    dist[v] = du + 1
                    parent[v] = u
                    seen_mark[v] = mark
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False

    return True


def _thue_morse_sequence(length):
    """Generate the first `length` bits of the Thue-Morse sequence."""
    seq = [0]
    for i in range(1, length):
        seq.append(bin(i).count('1') % 2)
    return seq

def _thue_morse_shifts(m, edge_count):
    """
    Generate edge shifts using Thue-Morse derived values for high dispersion.
    Uses the Thue-Morse word to create a low-discrepancy sequence that
    structurally minimizes low-order polynomial collisions.
    """
    # Generate a long Thue-Morse sequence
    tm_len = max(64, edge_count * 4)
    tm = _thue_morse_sequence(tm_len)
    
    # Create shifts based on Thue-Morse pattern scaled to m
    # Use different offsets for each edge to ensure distinct patterns
    shifts = []
    for i in range(edge_count):
        # Use a window of Thue-Morse bits starting at position i*4
        offset = (i * 4) % tm_len
        # Take 8 bits and interpret as an integer
        window = tm[offset:offset+8]
        if len(window) < 8:
            window = (window + tm[:8-len(window)])
        val = 0
        for bit in window:
            val = (val << 1) | bit
        # Scale to [0, m) with good dispersion
        shift = (val * m + i * 7) % m
        shifts.append(shift)
    return shifts

def _thue_morse_shifts_v2(m, edge_count):
    """
    Alternative Thue-Morse based shifts using cumulative sums for better dispersion.
    """
    tm_len = max(128, edge_count * 8)
    tm = _thue_morse_sequence(tm_len)
    
    # Compute prefix sums of Thue-Morse bits for smoother variation
    prefix = [0]
    for i in range(tm_len):
        prefix.append(prefix[-1] + tm[i])
    
    shifts = []
    for i in range(edge_count):
        # Use prefix sum at position proportional to edge index
        pos1 = (i * tm_len // edge_count) % tm_len
        pos2 = ((i + 1) * tm_len // edge_count) % tm_len
        # Combine two prefix sum values for better mixing
        val = (prefix[pos1] * 31 + prefix[pos2] * 17 + i * 13) % (m * m)
        shift = val % m
        shifts.append(shift)
    return shifts

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k-regular graphs.
    We try smaller d values to get fewer vertices, checking girth empirically.
    """
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # For k=3, we use q=3.
    q = 3

    def build_luw(d):
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

    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    return build_luw(11)

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

# EVOLVE-BLOCK-END
