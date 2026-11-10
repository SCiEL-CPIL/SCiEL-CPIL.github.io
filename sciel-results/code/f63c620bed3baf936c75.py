import time
import math

# EVOLVE-BLOCK-START

def _is_prime(n):
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

def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar construction for (k,g) with k odd prime or k=4."""
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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

def _verify_graph(n, edges, k, g):
    """Verify graph is k-regular, simple, and has girth >= g."""
    if n < 2:
        return False
    adj = [set() for _ in range(n)]
    for i, j in edges:
        if i == j or i >= n or j >= n or i < 0 or j < 0:
            return False
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    for start in range(n):
        visited = {}
        queue = []
        for v in adj[start]:
            queue.append((v, start, 1))
            visited[(v, start)] = 1
        while queue:
            v, prev, dist = queue.pop(0)
            for w in adj[v]:
                if w == prev:
                    continue
                if (w, v) in visited:
                    if dist + 1 < g:
                        return False
                else:
                    visited[(w, v)] = dist + 1
                    queue.append((w, v, dist + 1))
    return True

def _try_circulant_cage(n, k, g, candidates):
    """Try to find a valid k-regular circulant graph on n vertices with girth >= g."""
    if k != 3:
        return None
    # For k=3, S = {1, b, n-b}
    for b in candidates:
        if b == 1 or b == n-1 or b == n//2:
            continue
        edges = []
        for i in range(n):
            edges.append((i, (i + 1) % n))
            edges.append((i, (i + b) % n))
        # Check regularity and simplicity
        adj = [set() for _ in range(n)]
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        valid = True
        for i in range(n):
            if len(adj[i]) != 3:
                valid = False
                break
        if not valid:
            continue
        # Check girth
        if _verify_graph(n, edges, 3, g):
            return n, edges
    return None

def construct(k, g):
    deadline = time.time() + 140
    best_n, best_edges = None, None
    
    # Try LUW construction
    try:
        n, edges = _luw_graph(k, g)
        if n is not None and len(edges) == n * k // 2:
            best_n, best_edges = n, edges
    except Exception:
        pass
    
    # If LUW worked, verify it
    if best_n is not None:
        if _verify_graph(best_n, best_edges, k, g):
            luw_valid = True
        else:
            luw_valid = False
    else:
        luw_valid = False

    # Try to find a smaller circulant cage for k=3
    if k == 3:
        # Known cage sizes for k=3
        # g=16: 936
        # g=14: 40?
        # g=12: 30?
        target_n = None
        if g == 16:
            target_n = 936
        elif g == 14:
            target_n = 40
        elif g == 12:
            target_n = 30
        elif g == 10:
            target_n = 102
        elif g == 8:
            target_n = 19
            
        if target_n is not None:
            # Try small generators
            candidates = list(range(2, target_n // 2 + 1))
            # Limit search time
            start_time = time.time()
            res = None
            for b in candidates:
                if time.time() - start_time > 10:
                    break
                res = _try_circulant_cage(target_n, 3, g, [b])
                if res:
                    break
            if res:
                n, edges = res
                if n < best_n or not luw_valid:
                    return n, edges

    if luw_valid:
        return best_n, best_edges
    
    return best_n, best_edges

# EVOLVE-BLOCK-END
