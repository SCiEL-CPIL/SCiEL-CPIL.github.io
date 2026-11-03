"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def _luw_graph(k, g, d_override=None, q_override=None, rule_variant=0):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q).
    Returns (n, edges) if valid q-regular with girth >= g, else None.
    rule_variant: 0 = standard, 1 = shifted, 2 = reversed"""
    q = q_override if q_override is not None else k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    elif q == 2:
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: x * y
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q
    
    # Build rule for LUW construction - multiple variants
    if rule_variant == 0:
        # Standard rule
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    elif rule_variant == 1:
        # Shifted rule: offset by 1
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 4) % 4, c - (c - 4) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    elif rule_variant == 2:
        # Reversed: use (b, a) instead of (a, b)
        rule = {2: (1, 1), 3: (1, 2), 4: (2, 1)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            rule[c] = (rule[c][1], rule[c][0])
    else:
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
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield (1 - side, tuple(y))

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
    
    n = len(order)
    return n, edges

def _verify_girth_fast(n, edges, k, g, timeout=5):
    """Fast girth verification using BFS from each vertex with early termination.
    Returns True if girth >= g, False otherwise."""
    start_time = time.time()
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check for multiple edges and self-loops
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Girth check: BFS from each vertex, track shortest cycle
    # We only need to check cycles of length < g, so we can limit BFS depth.
    # Use a smarter approach: only need to go to depth floor((g-1)/2) from each side
    
    max_depth = (g - 1) // 2
    
    for s in range(n):
        if time.time() - start_time > timeout:
            return False
        
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def construct(k, g):
    """Portfolio search over explicit LUW-type constructions."""
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    best = None
    best_n = float('inf')
    
    def try_candidate(n, edges, expected_degree):
        nonlocal best, best_n
        if n >= best_n:
            return
        # Quick degree check
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(dg != expected_degree for dg in deg):
            return
        # Check for duplicate edges
        edge_set = set()
        for u, v in edges:
            key = (min(u,v), max(u,v))
            if key in edge_set:
                return
            edge_set.add(key)
        # Verify girth
        remaining = deadline - time.time() - 2.0
        if remaining < 1.0:
            return
        if _verify_girth_fast(n, edges, expected_degree, g, timeout=min(remaining, 8)):
            best = (n, edges)
            best_n = n
    
    # Standard d for LUW: d = g-5 for odd g, or g-4 for even g
    d_std = g - 5 if (g - 5) % 2 == 1 else g - 4
    d_std = max(3, d_std)
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Candidate (q, d, rule_variant) tuples to try
    candidates = []
    
    # Priority 1: q = k with various d values and rule variants
    for d in [d_std, d_std + 1, d_std + 2, d_std - 1, d_std + 3]:
        if d >= 3:
            for rv in [0, 1, 2]:
                candidates.append((k, d, rv))
    
    # Priority 2: try q = k+1, k+2, etc. if they are prime (for potential subgraph extraction)
    for q_val in range(k + 1, min(k + 30, 200)):
        if is_prime(q_val):
            for d in [d_std, d_std + 1]:
                if d >= 3:
                    candidates.append((q_val, d, 0))
    
    # Deduplicate and limit
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    candidates = unique_candidates[:20]  # Limit to avoid timeout
    
    # Try each candidate
    for q_val, d_val, rv in candidates:
        if time.time() - start_time > deadline - 5:
            break
        try:
            result = _luw_graph(k, g, d_override=d_val, q_override=q_val, rule_variant=rv)
            if result is not None:
                n, edges = result
                # The LUW graph CD(d,q) is q-regular.
                # We need k-regular. Only accept if q == k.
                if q_val == k:
                    try_candidate(n, edges, k)
        except Exception:
            continue
    
    if best is not None:
        return best
    
    # Fallback: return the standard LUW construction
    return _luw_graph(k, g)
# EVOLVE-BLOCK-END
