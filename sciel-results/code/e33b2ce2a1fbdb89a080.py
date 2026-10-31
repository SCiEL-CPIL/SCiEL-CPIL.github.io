"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
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
                    dq.append((w, v))
    return True

def _build_k33_lift(m, seed=0):
    """Build a voltage graph lift of K_{3,3} over Z_m.
    K_{3,3}: 6 vertices (0,1,2 in left, 3,4,5 in right), 9 edges, 3-regular, bipartite.
    Returns (n, adj, edges) or None if invalid."""
    rng = _random.Random(seed)
    
    # K_{3,3} edges: left {0,1,2}, right {3,4,5}
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    
    n = 6 * m
    adj = [set() for _ in range(n)]
    edges = []
    
    # Assign random voltages to each edge
    voltages = {}
    for u, v in k33_edges:
        voltages[(u, v)] = rng.randint(0, m - 1)
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    for u, v in k33_edges:
        vol = voltages[(u, v)]
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    
    # Check degree - each vertex should have degree exactly 3
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return n, adj, edges

def _build_petersen_lift(m, seed=0):
    """Build a voltage graph lift of the Petersen graph over Z_m.
    Returns (n, adj, edges) or None if invalid."""
    rng = _random.Random(seed)
    
    # Petersen graph: 10 vertices, 15 edges, 3-regular, girth 5
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    n = 10 * m
    adj = [set() for _ in range(n)]
    edges = []
    
    voltages = {}
    for u, v in petersen_edges:
        voltages[(u, v)] = rng.randint(0, m - 1)
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    for u, v in petersen_edges:
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
    
    return n, adj, edges

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if k != 3:
        # Fallback to LUW for other k values
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
        else:
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
    
    # For k=3: Use voltage graph lifts
    # Moore bound for k=3, girth g: 2*(2^(g/2)-1) for even g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    # Strategy 1: K_{3,3} lifts (n=6m, smaller base)
    # n = 6*m >= moore => m >= moore/6
    m_min_k33 = max(1, (moore + 5) // 6)
    
    # Try K_{3,3} lifts first since they have fewer vertices per m
    for m in range(m_min_k33, m_min_k33 + 300):
        if _time.time() > deadline - 5:
            break
        n = 6 * m
        if n < moore:
            continue
        
        max_attempts = 80
        for attempt in range(max_attempts):
            if _time.time() > deadline - 3:
                break
            result = _build_k33_lift(m, seed=attempt)
            if result is None:
                continue
            n, adj, edges = result
            if _verify_girth(adj, n, g):
                return n, edges
    
    # Strategy 2: Petersen lifts (n=10m)
    m_min_pet = max(1, (moore + 9) // 10)
    
    for m in range(m_min_pet, m_min_pet + 150):
        if _time.time() > deadline - 5:
            break
        n = 10 * m
        if n < moore:
            continue
        
        max_attempts = 50
        for attempt in range(max_attempts):
            if _time.time() > deadline - 3:
                break
            result = _build_petersen_lift(m, seed=attempt)
            if result is None:
                continue
            n, adj, edges = result
            if _verify_girth(adj, n, g):
                return n, edges
    
    # Fallback: LUW construction
    q = 3
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
# EVOLVE-BLOCK-END
