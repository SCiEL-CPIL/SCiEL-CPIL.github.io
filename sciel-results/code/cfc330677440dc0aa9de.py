import time
import collections
import math

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _luw_construct(k, g, d_override=None):
    """
    LUW construction for 3-regular graphs.
    """
    q = k
    if d_override is None:
        d = (g - 2) // 2 + 1
        if q % 2 == 1 and d % 2 == 0:
            d += 1
    else:
        d = d_override

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        if _time_left() < 10.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

def _check_girth(n, edges, min_girth):
    """
    Check if girth is at least min_girth.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = collections.deque([start])
        while queue:
            curr = queue.popleft()
            if dist[curr] >= min_girth // 2:
                continue
            for neighbor in adj[curr]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[curr] + 1
                    queue.append(neighbor)
                elif neighbor != (curr - 1 if curr > 0 else -1): # This check is wrong for general graph
                    # Need to track parent
                    pass
    
    # Correct BFS with parent tracking
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = collections.deque([start])
        while queue:
            curr = queue.popleft()
            if dist[curr] > min_girth // 2:
                continue
            for neighbor in adj[curr]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[curr] + 1
                    parent[neighbor] = curr
                    queue.append(neighbor)
                elif parent[curr] != neighbor:
                    cycle_len = dist[curr] + dist[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0

    # Try LUW with smaller d if possible
    # For g=16, d=8 is the target.
    # If the code increments to 9, try 8 first.
    d_target = (g - 2) // 2 + 1
    if k % 2 == 1 and d_target % 2 == 0:
        # Try d_target first
        n_try, edges_try = _luw_construct(k, g, d_override=d_target)
        if _check_girth(n_try, edges_try, g):
            return n_try, edges_try
        
    # Fallback to standard LUW (which might increment d)
    n_luw, edges_luw = _luw_construct(k, g)
    
    return n_luw, edges_luw
