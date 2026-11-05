import time
from collections import deque
import numpy as np
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if k == 3 and g == 16:
        # Deterministic algebraic construction: Petersen graph lifted over Z_p
        # with structured voltage assignments from quadratic residues
        best_n = float('inf')
        best_edges = None

        # Try primes in increasing order (smaller p = fewer vertices = better score)
        # n = 10 * p, so p=106 gives n=1060 (worse than 936)
        # p=93 gives n=930 (better than 936!)
        # Try primes from 93 upward
        primes_to_try = [93, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
        # Filter to actual primes
        def is_prime(n):
            if n < 2:
                return False
            if n % 2 == 0:
                return n == 2
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True
        primes_to_try = [p for p in primes_to_try if is_prime(p)]
        # Sort by n=10*p ascending
        primes_to_try.sort()

        for p in primes_to_try:
            if time.time() > deadline - 15:
                break
            n_total = 10 * p
            # Try multiple deterministic voltage patterns
            for pattern_idx in range(20):
                if time.time() > deadline - 15:
                    break
                edges = _build_petersen_lift(p, pattern_idx)
                if edges is None:
                    continue
                adj = [[] for _ in range(n_total)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                if _check_girth_fast(adj, n_total, g):
                    best_n = n_total
                    best_edges = edges
                    break
            if best_edges is not None:
                break

        if best_edges is not None:
            return best_n, best_edges

        # Fallback to LUW
        n_luw, edges_luw = _luw_graph(k, g)
        return n_luw, edges_luw

    # General fallback
    n_luw, edges_luw = _luw_graph(k, g)
    return n_luw, edges_luw

def _build_petersen_lift(p, pattern_idx):
    """Build Petersen graph lifted over Z_p with deterministic voltage assignments."""
    # Petersen graph: 10 vertices, 15 edges
    # Standard labeling: outer 5-cycle (0-1-2-3-4-0), inner 5-point star (5-7-9-6-8-5), spokes (0-5,1-6,2-7,3-8,4-9)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    
    # Generate deterministic voltages based on pattern_idx
    # Use different algebraic structures for different patterns
    voltages = {}
    for idx, (u, v) in enumerate(petersen_edges):
        if pattern_idx == 0:
            # Quadratic residue based
            voltages[(u, v)] = (idx * idx + 1) % p
        elif pattern_idx == 1:
            # Linear with offset
            voltages[(u, v)] = (idx * 7 + pattern_idx) % p
        elif pattern_idx == 2:
            # Use primitive root structure
            voltages[(u, v)] = (pow(2, idx, p) + 1) % p
        elif pattern_idx == 3:
            # Mixed
            voltages[(u, v)] = (idx * 3 + idx * idx) % p
        elif pattern_idx == 4:
            voltages[(u, v)] = (pow(3, idx, p)) % p
        elif pattern_idx == 5:
            voltages[(u, v)] = (idx * 11 + 5) % p
        elif pattern_idx == 6:
            voltages[(u, v)] = (pow(5, idx, p) + idx) % p
        elif pattern_idx == 7:
            voltages[(u, v)] = (idx * idx + idx + 1) % p
        elif pattern_idx == 8:
            voltages[(u, v)] = (pow(7, idx, p)) % p
        elif pattern_idx == 9:
            voltages[(u, v)] = (idx * 13 + idx * idx) % p
        elif pattern_idx == 10:
            voltages[(u, v)] = (pow(11, idx, p) + 3) % p
        elif pattern_idx == 11:
            voltages[(u, v)] = (idx * 17 + 7) % p
        elif pattern_idx == 12:
            voltages[(u, v)] = (pow(13, idx, p)) % p
        elif pattern_idx == 13:
            voltages[(u, v)] = (idx * idx * 2 + idx) % p
        elif pattern_idx == 14:
            voltages[(u, v)] = (pow(17, idx, p) + idx * 2) % p
        elif pattern_idx == 15:
            voltages[(u, v)] = (idx * 19 + idx * idx * 3) % p
        elif pattern_idx == 16:
            voltages[(u, v)] = (pow(19, idx, p)) % p
        elif pattern_idx == 17:
            voltages[(u, v)] = (idx * 23 + 1) % p
        elif pattern_idx == 18:
            voltages[(u, v)] = (pow(23, idx, p) + idx * idx) % p
        else:
            voltages[(u, v)] = (idx * 29 + idx * idx * 5) % p
        # Ensure no zero voltage (would create multiple edges or loops in some cases)
        if voltages[(u, v)] == 0:
            voltages[(u, v)] = 1
        voltages[(v, u)] = (-voltages[(u, v)]) % p

    # Build the lifted graph
    edges = []
    edge_set = set()
    
    for u, v in petersen_edges:
        vu = voltages[(u, v)]
        for i in range(p):
            u_lift = u * p + i
            v_lift = v * p + ((i + vu) % p)
            a, b = min(u_lift, v_lift), max(u_lift, v_lift)
            if (a, b) not in edge_set:
                edge_set.add((a, b))
                edges.append((a, b))
    
    # Verify 3-regular
    if len(edges) != 15 * p:
        return None
    
    return edges

def _check_girth_fast(adj, n, g_limit):
    # Use BFS from each vertex to check for cycles shorter than g_limit
    max_depth = (g_limit - 1) // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < max_depth:
                        queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g_limit:
                        return False
    return True

def _luw_graph(k, g):
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
