"""Cayley graph over GF(2) with algebraic girth check for n(3,16)."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _cayley_girth_check(d, generators):
    """Check girth of Cayley graph on (Z_2)^d with given generators (as bitmasks).
    Uses BFS from identity with proper cycle detection."""
    n = 1 << d
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    q = deque([0])
    min_cycle = float('inf')
    
    while q:
        u = q.popleft()
        for gen in generators:
            v = u ^ gen
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
            elif v != parent[u] and u != parent[v]:
                cycle_len = dist[u] + dist[v] + 1
                if cycle_len < min_cycle:
                    min_cycle = cycle_len
        if min_cycle <= 4:
            break
    
    return min_cycle if min_cycle < float('inf') else 999


def _find_gf2_generators(start_d, end_d, k, g, deadline):
    """Search for k generators in (Z_2)^d for d in [start_d, end_d)
    such that the Cayley graph has girth >= g.
    Returns (generators, d) or (None, None)."""
    best_d = None
    best_gens = None
    
    for d in range(start_d, end_d):
        if time.time() > deadline:
            break
        n = 1 << d
        
        # Number of trials scales with d
        num_trials = 300 if d <= 10 else 150
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Generate k random non-zero distinct vectors
            gens = []
            for _ in range(k):
                while True:
                    x = random.randint(1, n - 1)
                    if x not in gens:
                        gens.append(x)
                        break
            
            # Check girth
            girth = _cayley_girth_check(d, gens)
            if girth >= g:
                if best_d is None or d < best_d:
                    best_d = d
                    best_gens = gens
                break  # Found valid for this d, try smaller d
    
    if best_gens is not None:
        return best_gens, best_d
    return None, None


def _cayley_to_edges(d, generators):
    """Convert Cayley graph on (Z_2)^d with given generators to edge list."""
    n = 1 << d
    edges = []
    for i in range(n):
        for gen in generators:
            j = i ^ gen
            if i < j:
                edges.append((i, j))
    return edges


def construct(k, g):
    """Construct a k-regular graph of girth at least g with minimal vertices."""
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    if k != 3 or g != 16:
        # Fallback for other parameters
        return _lzw_fallback(k, g)
    
    # Strategy 1: Cayley graph over GF(2), try d from 9 upward
    random.seed(42)
    gens, d = _find_gf2_generators(9, 14, k, g, deadline)
    
    if gens is not None:
        n = 1 << d
        edges = _cayley_to_edges(d, gens)
        # Verify
        if _verify_girth(n, edges, g) and _check_regular(n, edges, k):
            return n, edges
    
    # Fallback: LZW construction
    return _lzw_fallback(k, g)


def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        dist = {}
        q = deque()
        for v in adj[start]:
            dist[(start, v)] = 1
            q.append((start, v))
        while q:
            prev, curr = q.popleft()
            d = dist[(prev, curr)]
            if d >= g:
                continue
            if curr == start and prev != -1:
                return False
            for nxt in adj[curr]:
                if nxt == prev:
                    continue
                state = (curr, nxt)
                if state not in dist:
                    dist[state] = d + 1
                    q.append(state)
    return True


def _check_regular(n, edges, k):
    """Check graph is simple and k-regular."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    return True


def _lzw_fallback(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
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

# EVOLVE-BLOCK-END
