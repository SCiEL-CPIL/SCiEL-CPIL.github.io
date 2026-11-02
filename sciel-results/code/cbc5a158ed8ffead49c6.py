import time
from collections import deque

_DEADLINE = 150.0

def construct(k, g):
    start = time.time()
    deadline = start + _DEADLINE
    
    # If k=4 and g=10, try the known 320-vertex Lubell graph
    if k == 4 and g == 10:
        try:
            n, edges = _construct_lubell_320()
            # Validate
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            valid = True
            for i in range(n):
                if len(adj[i]) != k:
                    valid = False
                    break
            
            if valid and not _has_short_cycle(adj, n, g):
                return n, edges
        except Exception:
            pass

    # Fallback to LUW
    q = k
    min_d = max(3, g // 2 - 1)
    for d in range(min_d, 10):
        if time.time() > deadline - 5:
            break
        n, edges = _luw(k, g, d)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        valid = True
        for i in range(n):
            if len(adj[i]) != k:
                valid = False
                break
                
        if valid and not _has_short_cycle(adj, n, g):
            return n, edges
            
    n, edges = _luw(k, g, 9)
    return n, edges

def _construct_lubell_320():
    # Cayley graph on Z2 x Z2 x Z80
    # Generators: (1,0,1), (0,1,1), (1,0,-1), (0,1,-1)
    n1, n2, n3 = 2, 2, 80
    n = n1 * n2 * n3
    gens = [(1, 0, 1), (0, 1, 1), (1, 0, -1), (0, 1, -1)]
    
    def add(x, y):
        return ((x[0]+y[0])%n1, (x[1]+y[1])%n2, (x[2]+y[2])%n3)
        
    def sub(x, y):
        return ((x[0]-y[0])%n1, (x[1]-y[1])%n2, (x[2]-y[2])%n3)
        
    # Map group elements to indices 0..319
    # Index = i1 * (n2*n3) + i2 * n3 + i3
    index_map = {}
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                index_map[(i1, i2, i3)] = i1 * (n2 * n3) + i2 * n3 + i3
                
    edges = []
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                v = (i1, i2, i3)
                idx_v = index_map[v]
                for g in gens:
                    w = add(v, g)
                    idx_w = index_map[w]
                    if idx_v < idx_w:
                        edges.append((idx_v, idx_w))
                        
    return n, edges

def _has_short_cycle(adj, n, g):
    for s in range(n):
        frontier = deque()
        for nb in adj[s]:
            frontier.append((nb, s))
        depth = 1
        while frontier:
            if depth + 1 < g:
                for node, prev in frontier:
                    if prev != s:
                        for nb in adj[node]:
                            if nb == s:
                                return True
            if depth >= g - 2:
                break
            nxt = deque()
            while frontier:
                node, prev = frontier.popleft()
                for nb in adj[node]:
                    if nb != prev:
                        nxt.append((nb, node))
            frontier = nxt
            depth += 1
    return False

def _luw(k, g, d=None):
    q = k
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
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
