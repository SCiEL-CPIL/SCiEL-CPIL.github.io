"""Rewritten program for (3,16)-cage construction with improved verification and search."""
import time as _time
import random as _random
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

def _verify_girth_array(adj_list, n, g):
    """Faster verification using array-based distance tracking."""
    for start in range(n):
        visited = [-1] * (n * (n + 1))
        dq = deque()
        enc_start = start * (n + 1) + 0
        visited[enc_start] = 0
        dq.append((start, -1, 0))
        while dq:
            v, prev, d = dq.popleft()
            if d >= g - 1:
                continue
            for w in adj_list[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                enc = w * (n + 1) + (v + 1)
                if visited[enc] == -1:
                    visited[enc] = d + 1
                    dq.append((w, v, d + 1))
    return True

def _luw_construct(q, g):
    """LUW construction for (q+1)-regular graphs. Returns (n, edges)."""
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

def _get_petersen_edges():
    """Return Petersen graph edges and vertex count."""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    return edges, 10

def _build_lift_petersen(m, voltages):
    """Build a voltage graph lift of the Petersen graph. Returns (n, adj_list, edges) or None."""
    base_edges, n_base = _get_petersen_edges()
    n = n_base * m
    adj = [set() for _ in range(n)]
    edges = []
    
    for u, v in base_edges:
        vol = voltages[(u, v)]
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
    
    adj_list = [sorted(adj[i]) for i in range(n)]
    return n, adj_list, edges

def _is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _find_primitive_roots(m):
    """Find all primitive roots modulo m. Returns list of primitive roots."""
    if m <= 1 or not _is_prime(m):
        return []
    phi = m - 1
    factors = []
    temp = phi
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.append(temp)
    
    roots = []
    for g in range(2, m):
        is_primitive = True
        for p in factors:
            if pow(g, phi // p, m) == 1:
                is_primitive = False
                break
        if is_primitive:
            roots.append(g)
    return roots

def _thue_morse(n):
    """Compute Thue-Morse sequence up to n terms."""
    tm = [0]
    for i in range(1, n):
        tm.append(tm[i >> 1] ^ (i & 1))
    return tm

def _thue_morse_shifts(m, max_shifts=30):
    """Generate Thue-Morse derived voltage patterns."""
    tm = _thue_morse(max(m * 2, 100))
    patterns = []
    for shift in range(max_shifts):
        pattern = [tm[(i + shift) % len(tm)] for i in range(m)]
        patterns.append(pattern)
    return patterns

def _cubic_shifts(m, max_shifts=20):
    """Generate cubic polynomial voltage patterns."""
    patterns = []
    for a in range(1, min(5, m) + 1):
        for b in range(0, min(3, m)):
            for c in range(0, min(3, m)):
                pattern = [(a * i * i * i + b * i * i + c * i) % m for i in range(m)]
                patterns.append(pattern)
    return patterns

def _fibonacci_shifts(m):
    """Generate Fibonacci-based voltage patterns."""
    patterns = []
    for seed in range(1, min(10, m) + 1):
        pattern = []
        a, b = 0, seed
        for i in range(m):
            pattern.append(b % m)
            a, b = b, (a + b) % m
        patterns.append(pattern)
    return patterns

def _try_voltage_pattern(base_edges, n_base, m, g, pattern):
    """Try a specific voltage pattern on the base graph."""
    voltages = {}
    for i, (u, v) in enumerate(base_edges):
        vol = pattern[i % m] % m
        voltages[(u, v)] = vol
        voltages[(v, u)] = (-vol) % m
    
    result = _build_lift_petersen(m, voltages)
    if result is None:
        return None
    n, adj_list, edges = result
    if _verify_girth_array(adj_list, n, g):
        return n, edges
    return None

def _search_voltage_lifts(m_min, m_max, g, deadline, best_n):
    """Search for valid voltage lifts in the range [m_min, m_max]."""
    base_edges, n_base = _get_petersen_edges()
    best_result = None
    best_n_local = best_n
    
    for m in range(m_min, m_max + 1):
        if _time.time() > deadline - 10:
            break
        n_lift = n_base * m
        if n_lift < best_n_local:
            # Try Thue-Morse patterns
            for pattern in _thue_morse_shifts(m, max_shifts=15):
                if _time.time() > deadline - 10:
                    break
                result = _try_voltage_pattern(base_edges, n_base, m, g, pattern)
                if result is not None:
                    n_r, edges_r = result
                    if n_r < best_n_local:
                        best_n_local = n_r
                        best_result = (n_r, edges_r)
                    break
            
            # Try cubic patterns
            if best_n_local > n_lift:
                for pattern in _cubic_shifts(m, max_shifts=10):
                    if _time.time() > deadline - 10:
                        break
                    result = _try_voltage_pattern(base_edges, n_base, m, g, pattern)
                    if result is not None:
                        n_r, edges_r = result
                        if n_r < best_n_local:
                            best_n_local = n_r
                            best_result = (n_r, edges_r)
                        break
            
            # Try Fibonacci patterns
            if best_n_local > n_lift:
                for pattern in _fibonacci_shifts(m):
                    if _time.time() > deadline - 10:
                        break
                    result = _try_voltage_pattern(base_edges, n_base, m, g, pattern)
                    if result is not None:
                        n_r, edges_r = result
                        if n_r < best_n_local:
                            best_n_local = n_r
                            best_result = (n_r, edges_r)
                        break
            
            # Try random patterns
            if best_n_local > n_lift:
                rng = _random.Random(m * 7919)
                for attempt in range(30):
                    if _time.time() > deadline - 10:
                        break
                    pattern = [rng.randint(0, m - 1) for _ in range(m)]
                    result = _try_voltage_pattern(base_edges, n_base, m, g, pattern)
                    if result is not None:
                        n_r, edges_r = result
                        if n_r < best_n_local:
                            best_n_local = n_r
                            best_result = (n_r, edges_r)
                        break
    
    return best_result, best_n_local

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115
    
    if k != 3:
        n, edges = _luw_construct(k, g)
        return n, edges
    
    # Moore bound for k=3, girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    # LUW construction as a strong baseline
    n_luw, edges_luw = _luw_construct(3, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)
    
    # Search for better voltage lifts
    m_min = max(1, (moore + 9) // 10)
    m_max = m_min + 200
    
    result, best_n = _search_voltage_lifts(m_min, m_max, g, deadline, best_n)
    if result is not None:
        best_result = result
    
    return best_result
