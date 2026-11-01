"""Search for smallest 4-regular graph of girth >= g using circulants + random fallback."""
import time
import random
from collections import deque
from math import gcd

_DEADLINE = None

def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0

def _check_deadline():
    return time.time() > _DEADLINE

def _girth_bfs(n, adj):
    """Exact girth via BFS from each vertex."""
    best = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            v = q.popleft()
            for nv in adj[v]:
                if dist[nv] == -1:
                    dist[nv] = dist[v] + 1
                    parent[nv] = v
                    q.append(nv)
                elif parent[v] != nv and parent[nv] != v:
                    cycle_len = dist[v] + dist[nv] + 1
                    if cycle_len < best:
                        best = cycle_len
    return best

def _circulant_girth(n, S):
    """Girth of circulant C(n; S) via non-backtracking BFS (faster for circulants)."""
    Sset = set(S)
    best = float('inf')
    for start in range(n):
        # BFS over (vertex, prev) states
        visited = {}
        queue = deque()
        for s in Sset:
            nxt = (start + s) % n
            state = (nxt, start)
            if state not in visited:
                visited[state] = 1
                queue.append(state)
        while queue:
            v, pv = queue.popleft()
            depth = visited[(v, pv)]
            for s in Sset:
                nv = (v + s) % n
                if nv == pv:
                    continue
                if nv == start and depth + 1 >= 3:
                    if depth + 1 < best:
                        best = depth + 1
                    continue
                state = (nv, v)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append(state)
    return best

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (seed fallback)."""
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

def construct(k, g):
    _init_deadline()
    
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    # Search circulants C(n; {a,-a,b,-b}) for smallest n with girth >= 10
    best_n = None
    best_S = None
    
    # Moore bound lower bound is 243, known best is 320
    # Search n from 243 upward
    for n in range(243, 500):
        if _check_deadline():
            break
        if best_n is not None and n >= best_n:
            break
        # S = {a, -a, b, -b}
        for a in range(1, n // 2 + 1):
            if _check_deadline():
                break
            if 2 * a % n == 0:
                continue
            for b in range(a + 1, n // 2 + 1):
                if _check_deadline():
                    break
                if 2 * b % n == 0:
                    continue
                if (a + b) % n == 0:
                    continue
                if gcd(n, a, b) != 1:
                    continue
                S = (a, n - a, b, n - b)
                girth = _circulant_girth(n, S)
                if girth >= g:
                    if best_n is None or n < best_n:
                        best_n = n
                        best_S = S
                    break
        if best_n is not None and n >= best_n:
            break
    
    if best_n is not None:
        n = best_n
        S = best_S
        edges = []
        Sset = set(S)
        for i in range(n):
            for s in Sset:
                j = (i + s) % n
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # Fallback: L-U-W
    return _luw_construct(k, g)
