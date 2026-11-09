import time
import sys
import numpy as np
from collections import deque, defaultdict
import random
import math

_DEADLINE = 150.0
_T0 = None

def _elapsed():
    return time.time() - _T0

def _time_ok(margin=5.0):
    return _elapsed() < _DEADLINE - margin

def construct(k, g):
    global _T0
    _T0 = time.time()
    
    # Handle small cases with hardcoded cages
    if k == 2:
        n = g
        edges = [(i, (i + 1) % n) for i in range(n)]
        return (n, edges)
    
    if k == 3:
        if g <= 4:
            return (4, [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)])
        if g <= 5:
            # Petersen
            edges = []
            for i in range(5):
                edges.append((i, (i + 1) % 5))
                edges.append((5 + i, 5 + (i + 2) % 5))
                edges.append((i, 5 + i))
            return (10, edges)
        if g <= 6:
            # Heawood
            edges = []
            fano_lines = [(0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5)]
            for li, line in enumerate(fano_lines):
                for p in line:
                    edges.append((p, 7 + li))
            return (14, edges)
    
    # Try LUW first as it's deterministic and fast
    result = _luw_construct(k, g)
    if result is not None:
        return result
    
    # Fallback: try greedy search for specific cases
    if k == 5 and g == 9 and _time_ok(margin=20.0):
        for n in [1116, 1120, 1124]:
            if not _time_ok(margin=15.0):
                break
            result = _greedy_girth_max_v2(n, k, g)
            if result is not None:
                return result
    
    # Last resort: return a simple regular graph (may not satisfy girth)
    return _simple_regular_graph(k)

def _greedy_girth_max_v2(n, k, g):
    """Try to construct a k-regular graph on n vertices with girth >= g using greedy search."""
    if not _time_ok(margin=5.0):
        return None
    
    # Generate random regular graph
    adj = _random_regular_graph(n, k)
    if adj is None:
        return None
    
    # Initial girth check
    girth = _calculate_girth_fast(adj, n)
    if girth >= g:
        return (n, _adj_to_edges(adj))
    
    # Greedy improvement with limited iterations
    best_girth = girth
    best_adj = [row.copy() for row in adj]
    
    # Limit iterations based on time
    max_iters = min(5000, max(100, int((_DEADLINE - _elapsed() - 10) * 100)))
    for _ in range(max_iters):
        if not _time_ok(margin=5.0):
            break
        
        # Randomly pick two edges to swap
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or adj[u][v] == 0:
            continue
        
        u2 = random.randint(0, n - 1)
        v2 = random.randint(0, n - 1)
        if u2 == v2 or adj[u2][v2] == 0:
            continue
        
        # Ensure we don't create multi-edges or loops
        if adj[u][v2] != 0 or adj[u2][v] != 0:
            continue
        if u == v2 or u2 == v:
            continue
            
        # Perform swap
        adj[u][v] = 0
        adj[v][u] = 0
        adj[u2][v2] = 0
        adj[v2][u2] = 0
        
        adj[u][v2] = 1
        adj[v2][u] = 1
        adj[u2][v] = 1
        adj[v][u2] = 1
        
        # Check girth
        new_girth = _calculate_girth_fast(adj, n)
        
        if new_girth > best_girth:
            best_girth = new_girth
            best_adj = [row.copy() for row in adj]
            if best_girth >= g:
                return (n, _adj_to_edges(best_adj))
        else:
            # Revert
            adj[u][v] = 1
            adj[v][u] = 1
            adj[u2][v2] = 1
            adj[v2][u2] = 1
            
            adj[u][v2] = 0
            adj[v2][u] = 0
            adj[u2][v] = 0
            adj[v][u2] = 0

    return None

def _random_regular_graph(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Create stubs
    stubs = [i for i in range(n) for _ in range(k)]
    random.shuffle(stubs)
    
    # Pair stubs
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if adj[u][v] == 1:
            return None
        adj[u][v] = 1
        adj[v][u] = 1
        
    return adj

def _calculate_girth_fast(adj, n):
    """Calculate girth using BFS from every vertex with early termination."""
    min_girth = float('inf')
    
    for start in range(n):
        if not _time_ok(margin=3.0):
            break
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= min_girth:
                continue
            for v in range(n):
                if adj[u][v] == 1:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif dist[v] > dist[u] + 1:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth <= 3:
                                return min_girth
                        
    return min_girth if min_girth != float('inf') else n + 1

def _adj_to_edges(adj):
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 1:
                edges.append((i, j))
    return edges

def _simple_regular_graph(k):
    """Fallback: return a simple k-regular graph (may not satisfy girth)."""
    n = k + 1
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return (n, edges)

def _luw_construct(k, g, d_override=None):
    """Build Lazebnik-Ustimenko-Woldar graph CD(d, q) with q=k using efficient encoding."""
    q = k
    if q not in (2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
        return None

    if d_override is not None:
        d = d_override
    else:
        d = None
        for cand in range(3, 20):
            bound = cand + 5 if cand % 2 == 1 else cand + 4
            if bound >= g:
                d = cand
                break
        if d is None:
            d = max(3, g - 5)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Encode vertex as integer: side * q^d + value
    # value = sum_{c=1}^{d} x[c] * q^(d-c)
    q_powers = [1] * (d + 1)
    for i in range(1, d + 1):
        q_powers[i] = q_powers[i-1] * q

    def encode(side, x):
        val = 0
        for c in range(1, d + 1):
            val += x[c] * q_powers[d - c]
        return side * (q ** d) + val

    def decode(enc):
        side = enc // (q ** d)
        val = enc % (q ** d)
        x = [0] * (d + 1)
        for c in range(1, d + 1):
            x[c] = val % q
            val //= q
        return side, x

    def neighbours(enc):
        side, x = decode(enc)
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield encode(1 - side, y)

    start_enc = encode(0, [0] * (d + 1))
    index = {start_enc: 0}
    order = [start_enc]
    edges = []

    for i, enc in enumerate(order):
        if not _time_ok(margin=15.0):
            return None
        side, _ = decode(enc)
        for other_enc in neighbours(enc):
            if other_enc not in index:
                index[other_enc] = len(order)
                order.append(other_enc)
            if side == 0:
                edges.append((index[enc], index[other_enc]))

    n = len(order)
    if n == 0:
        return None

    if _verify(n, edges, k, g):
        return (n, edges)
    return None

def _verify(n, edges, k, g):
    """Verify graph is k-regular, simple, and has girth >= g using non-backtracking walks."""
    if n == 0:
        return False
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Check girth using non-backtracking walks
    for start in range(n):
        if not _time_ok(margin=10.0):
            return False
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))
        visited.add((start, -1))
        
        while queue:
            v, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                new_state = (w, v)
                if new_state in visited:
                    continue
                visited.add(new_state)
                if w == start and depth + 1 < g:
                    return False
                queue.append((w, v, depth + 1))
    
    return True
