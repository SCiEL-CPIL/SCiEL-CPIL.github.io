"""Construct a k-regular graph with girth at least g on as few vertices as possible."""
import time
import random
from collections import deque

_DEADLINE = None

def _check_deadline():
    if _DEADLINE is not None and time.time() > _DEADLINE:
        return True
    return False

def _verify_girth(n, edges, k, g):
    """Verify the graph is k-regular, simple, and has girth >= g."""
    # Check degree
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u,v), max(u,v))
        if key in seen:
            return False
        seen.add(key)
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != k:
            return False
    # Check girth via BFS from each vertex (non-backtracking)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        # BFS: state = (vertex, prev_vertex)
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
        while queue:
            v, prev, dist = queue.popleft()
            if dist >= g - 1:
                # Found a cycle of length dist+1 >= g, which is fine
                # But we need to check if there's a shorter cycle
                pass
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == start and dist + 1 >= g:
                    # Cycle of length dist+1, which is >= g, OK
                    continue
                if nb == start and dist + 1 < g:
                    return False
                if (nb, v) not in visited:
                    visited.add((nb, v))
                    queue.append((nb, v, dist + 1))
    return True

def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q
    
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    
    start = (0, (None,) + (0,)*d)
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
    return len(order), edges

def _circulant_girth_ok(n, a, b, g):
    """Check if circulant C(n; a, b) has girth >= g using single BFS."""
    steps = [a, -a, b, -b]
    visited = [[-1]*4 for _ in range(n)]
    queue = deque()
    for si, step in enumerate(steps):
        pos = step % n
        if pos == 0:
            return False
        visited[pos][si] = 1
        queue.append((pos, si, 1))
    
    while queue:
        pos, si, dist = queue.popleft()
        if dist >= g - 1:
            continue
        for ni, step in enumerate(steps):
            if step == -steps[si]:
                continue
            new_pos = (pos + step) % n
            if new_pos == 0:
                if dist + 1 < g:
                    return False
                else:
                    continue
            if visited[new_pos][ni] == -1:
                visited[new_pos][ni] = dist + 1
                queue.append((new_pos, ni, dist + 1))
    return True

def _try_circulant(k, g, max_n=400):
    """Try to find a circulant graph with girth >= g."""
    if k % 2 != 0:
        return None
    best = None
    for n in range(k+1, max_n+1):
        if _check_deadline():
            break
        for a in range(1, n//2 + 1):
            if _check_deadline():
                break
            for b in range(a+1, n//2 + 1):
                if a == n-a or b == n-b or a == n-b or b == n-a:
                    continue
                if _circulant_girth_ok(n, a, b, g):
                    edges = set()
                    for i in range(n):
                        for step in [a, b]:
                            j = (i + step) % n
                            edges.add((min(i,j), max(i,j)))
                    edges = list(edges)
                    if best is None or n < best[0]:
                        best = (n, edges)
                        if n <= 250:
                            return best
    return best

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.time() + 140  # Leave 10s margin before 150s kill
    
    best = None
    
    # Try LUW graph first
    try:
        n_luw, edges_luw = _luw_graph(k, g)
        if _verify_girth(n_luw, edges_luw, k, g):
            best = (n_luw, edges_luw)
    except Exception:
        pass
    
    # Try circulant graphs if k is even
    if k % 2 == 0:
        try:
            circ = _try_circulant(k, g, max_n=400)
            if circ is not None:
                n_c, edges_c = circ
                if best is None or n_c < best[0]:
                    best = (n_c, edges_c)
        except Exception:
            pass
    
    if best is None:
        # Fallback: return LUW even if verification failed (shouldn't happen)
        try:
            n_luw, edges_luw = _luw_graph(k, g)
            return n_luw, edges_luw
        except Exception:
            # Last resort: trivial graph (will likely fail validation)
            n = k + 1
            edges = [(i, (i+1) % n) for i in range(n)]
            return n, edges
    
    return best
