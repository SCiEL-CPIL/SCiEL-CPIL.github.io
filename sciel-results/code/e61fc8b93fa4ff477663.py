import time
import collections
import random
import math

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _has_short_cycle(n, edges, min_girth):
    """
    Returns True if the graph has a cycle shorter than min_girth.
    Uses non-backtracking BFS from each vertex.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = min_girth // 2
    
    for start in range(n):
        # State: (current_vertex, previous_vertex)
        # dist[state] = depth
        dist = {}
        queue = collections.deque()
        
        for nb in adj[start]:
            state = (nb, start)
            dist[state] = 1
            queue.append(state)
        
        while queue:
            curr, prev = queue.popleft()
            d = dist[(curr, prev)]
            if d >= max_depth:
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if nb == start:
                    # Found a cycle of length d+1
                    if d + 1 < min_girth:
                        return True
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = d + 1
                    queue.append(state)
    return False

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # Search for Generalized Petersen Graph GP(n, k_gp)
    # N = 2n vertices.
    # Moore bound for k=3, g=16 is 510. So n >= 255.
    
    min_n = 255
    max_n = 800  # Extended search range
    
    # Try larger n first (more likely to have high girth), then smaller
    # Actually try in increasing order to find smallest n first
    # But use a smarter order: try n values that are likely to work
    
    # Quick modular check: for GP(n, k), the inner cycle structure
    # requires that k^j not ≡ ±1 (mod n) for j = 1, ..., g/2 - 1
    # Also need gcd(k, n) = 1 for the inner graph to be a single cycle
    
    candidates = []
    for n in range(min_n, max_n + 1):
        if _time_left() < 20.0:
            break
        for k_gp in range(1, n // 2 + 1):
            if k_gp == n - k_gp:
                continue
            # Must be coprime for inner to be one cycle
            if math.gcd(k_gp, n) != 1:
                continue
            # Quick modular check for girth
            valid = True
            val = k_gp % n
            for j in range(1, g // 2):
                val = (val * k_gp) % n
                if val == 1 or val == n - 1:
                    valid = False
                    break
            if valid:
                candidates.append((n, k_gp))
    
    # Sort by n (prefer fewer vertices)
    candidates.sort(key=lambda x: x[0])
    
    for n, k_gp in candidates:
        if _time_left() < 10.0:
            break
        
        edges = []
        # Outer
        for i in range(n):
            edges.append((i, (i + 1) % n))
        # Spokes
        for i in range(n):
            edges.append((i, n + i))
        # Inner
        for i in range(n):
            edges.append((n + i, n + (i + k_gp) % n))
            
        total_vertices = 2 * n
        
        if not _has_short_cycle(total_vertices, edges, g):
            return total_vertices, edges
    
    # Fallback to LUW
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
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
