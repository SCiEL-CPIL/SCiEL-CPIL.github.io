import time
import collections
import math

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _check_girth_bfs(n, edges, min_girth):
    """
    Checks if the graph has girth at least min_girth.
    Uses BFS from each vertex to find shortest cycle.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # For each vertex, BFS to find shortest cycle through it
    # Actually, checking from each edge is better for girth
    # But for small n, vertex BFS is fine.
    # Optimization: If we find a cycle of length L, we can stop if L >= min_girth?
    # No, we need to ensure NO cycle is < min_girth.
    
    max_depth = min_girth // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = collections.deque([start])
        
        while queue:
            curr = queue.popleft()
            if dist[curr] > max_depth:
                continue
            for neighbor in adj[curr]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[curr] + 1
                    parent[neighbor] = curr
                    queue.append(neighbor)
                elif neighbor != parent[curr]:
                    # Found a cycle
                    cycle_len = dist[curr] + dist[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _construct_gp(n, k):
    """
    Construct GP graph GP(n, k).
    Returns (2n, edges).
    """
    edges = []
    # Cycle 1: 0..n-1
    for i in range(n):
        edges.append((i, (i + 1) % n))
    # Cycle 2: n..2n-1
    for i in range(n):
        edges.append((n + i, n + (i + 1) % n))
    # Matching
    for i in range(n):
        edges.append((i, n + i))
    # Twisted Matching
    for i in range(n):
        edges.append((i, n + (i + k) % n))
        
    return 2 * n, edges

def _construct_luw(q, d):
    """
    Constructs the LUW graph CD(d, q).
    """
    if q == 3:
        add = lambda x, y: (x + y) % 3
        sub = lambda x, y: (x - y) % 3
        mul = lambda x, y: (x * y) % 3
    elif q == 4:
        table_add = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        table_mul = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add = lambda x, y: table_add[x][y]
        sub = lambda x, y: table_add[x][y]
        mul = lambda x, y: table_mul[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

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
        if _time_left() < 5.0:
            return None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0

    # Strategy 1: GP Graph Search
    # Moore bound for k=3, g=16 is ~510.
    # GP graph has 2n vertices. So 2n >= 510 => n >= 255.
    # Best known is 936 => n=468.
    # Search n from 255 to 1000.
    
    if k == 3:
        min_n_gp = 255
        max_n_gp = 1000
        
        for n in range(min_n_gp, max_n_gp + 1):
            if _time_left() < 10.0:
                break
            
            for k_gp in range(1, n):
                if _time_left() < 2.0:
                    break
                
                if math.gcd(n, k_gp) != 1:
                    continue
                
                # Quick filter: k^j != 1, -1 mod n for j < g/2
                valid = True
                val = 1
                for j in range(1, g // 2):
                    val = (val * k_gp) % n
                    if val == 1 or val == n - 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Construct and verify
                n_vertices, edges = _construct_gp(n, k_gp)
                
                # Check degree
                deg = [0] * n_vertices
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if any(dg != k for dg in deg):
                    continue
                
                # Check girth
                if _check_girth_bfs(n_vertices, edges, g):
                    return n_vertices, edges

    # Strategy 2: LUW Fallback
    q = k
    min_d = max(3, (g - 2) // 2)
    
    for d in range(min_d, min_d + 5):
        if _time_left() < 5.0:
            break
        
        result = _construct_luw(q, d)
        if result is None:
            continue
        
        n, edges = result
        
        # Quick degree check
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(dg != k for dg in deg):
            continue
        
        # Verify girth
        if _check_girth_bfs(n, edges, g):
            return n, edges
            
    # Absolute fallback
    n = 2 * k * 10
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
    for i in range(n):
        edges.append((i, (i + n // 2) % n))
    return n, edges
