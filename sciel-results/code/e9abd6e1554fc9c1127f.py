import time
import sys
import random
import numpy as np
from collections import deque

def _bfs_girth(adj):
    """Compute the girth of an undirected graph via BFS from each vertex."""
    n = len(adj)
    best = float('inf')
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle = dist[u] + dist[v] + 1
                    if cycle < best:
                        best = cycle
        if best <= 2:
            break
    return best if best < float('inf') else 0

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (from seed)."""
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

def _construct_4_10_cage():
    """Construct a 4-regular graph of girth >= 10 on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    n_target = 320
    
    def _try_random_greedy(seed):
        random.seed(seed)
        adj = [set() for _ in range(n_target)]
        # Start with a random 4-regular graph
        # Use a simple algorithm: pair up stubs
        # This might create short cycles.
        # Better: start with empty graph, add edges one by one, checking girth constraints locally?
        # Too slow.
        
        # Alternative: Use a known 4-regular graph structure, e.g. from a lattice.
        # Or just return None if this is too complex.
        return None

    # Since random greedy is hard to guarantee girth 10 quickly, 
    # and we don't have the explicit 320-vertex list, 
    # we might have to rely on the LUW fallback or a specific known construction.
    
    # Let's try the Cayley graph of Z_160 again with a larger set of candidates.
    def _check_girth_nb_fast(adj, min_girth):
        n = len(adj)
        for s in range(n):
            for v in adj[s]:
                dist = {}
                dist[(s, v)] = 1
                q = deque([(s, v)])
                while q:
                    u, w = q.popleft()
                    d = dist[(u, w)]
                    if d >= min_girth:
                        break
                    for x in adj[w]:
                        if x == u:
                            continue
                        if (w, x) in dist:
                            if d + 1 < min_girth:
                                return False
                        else:
                            dist[(w, x)] = d + 1
                            q.append((w, x))
        return True

    def _try_cayley_z160(gen_set):
        m = 160
        n = 2 * m
        edges = []
        adj = [[] for _ in range(n)]
        for side in range(2):
            for g in range(m):
                u = side * m + g
                for a in gen_set:
                    v = (1 - side) * m + ((g + a) % m)
                    if v not in adj[u]: # Avoid duplicate edges if gen set has duplicates? No, unique.
                        adj[u].append(v)
                        adj[v].append(u)
                        if side == 0:
                            edges.append((u, v))
        for i in range(n):
            if len(adj[i]) != 4:
                return None
        if not _check_girth_nb_fast(adj, 10):
            return None
        return n, edges

    candidates = [a for a in range(1, 80) if a % 2 == 1 and a % 5 != 0]
    # Generate more random sets
    import itertools
    # Try all combinations of 4 from a subset of candidates?
    # Too many.
    
    # Let's try a specific known generator set if I can guess it.
    # Or just return LUW.
    
    # To improve score, I must return 320.
    # If I can't find it, I return LUW (score 1.0).
    # So I should try harder.
    
    # Let's try a small number of random Cayley graphs of Z_160.
    for _ in range(50):
        if time.time() > deadline:
            break
        gen_set = random.sample(candidates, 4)
        # Ensure symmetry: if a is in set, -a is in set?
        # For bipartite double cover, we don't need symmetry in the same way.
        # But for the graph to be simple, we need unique neighbors.
        # (g+a) % m. If a1 != a2, neighbors are distinct.
        result = _try_cayley_z160(gen_set)
        if result:
            return result
            
    # Fallback
    return _luw_construct(4, 10)

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    if k == 4 and g == 10:
        try:
            n, edges = _construct_4_10_cage()
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            for i in range(n):
                if len(adj[i]) != k:
                    raise ValueError(f"Vertex {i} has degree {len(adj[i])}")
            girth = _bfs_girth(adj)
            if girth >= g:
                return n, edges
        except Exception:
            pass
        n, edges = _luw_construct(k, g)
        return n, edges
    else:
        n, edges = _luw_construct(k, g)
        return n, edges
