import time
import sys
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = 115.0

    def certify_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        N2 = n * n
        INF = g
        dist = [INF] * N2
        q = deque()

        for a in range(n):
            for b in adj[a]:
                dist[a * n + b] = 1
                q.append(a * n + b)

        while q:
            s = q.popleft()
            d = dist[s]
            if d >= g - 1:
                continue
            p = s // n
            c = s % n
            rev = c * n + p
            dr = dist[rev]
            if dr != INF:
                if d + dr + 1 < g:
                    return False
            for nb in adj[c]:
                if nb == p:
                    continue
                ns = c * n + nb
                nd = d + 1
                if nd < dist[ns]:
                    dist[ns] = nd
                    q.append(ns)
        return True

    def bosak_construct(k, g):
        # Bosak-type construction for cubic graphs of large girth
        # Vertices: (i, a, b) where i in [0, L), a,b in {0,1,2}
        # Three types of edges based on the index i
        # This produces a 3-regular graph
        
        # Try various L values to find one with sufficient girth
        # The graph has 12*L vertices
        
        # For k=3, g=16, we need a specific construction
        # Known: the (3,16)-cage has 936 vertices (best known)
        # Bosak graphs can achieve girth up to certain limits
        
        # Standard Bosak construction:
        # Vertices: V = {0,...,L-1} x {0,1,2} x {0,1,2}
        # For each (i, a, b):
        #   Edge 1: (i, a, b) -- (i+1 mod L, a', b')  [type A edge]
        #   Edge 2: (i, a, b) -- (i, a', b')          [type B edge, within same i]
        #   Edge 3: (i, a, b) -- (i-1 mod L, a'', b'') [type C edge]
        
        # The specific permutations determine the girth
        # For optimal girth, use specific Latin square based rules
        
        L = 4
        n = 12 * L
        edges = []
        
        # Define the edge structure for Bosak graph
        # Type 0 edges: connect (i, a, b) to (i+1 mod L, perm0(a), perm0(b))
        # Type 1 edges: connect (i, a, b) to (i, perm1(a), perm1(b))  
        # Type 2 edges: connect (i, a, b) to (i-1 mod L, perm2(a), perm2(b))
        
        # For L=4, use specific permutations
        # perm0: (a,b) -> (a+1 mod 3, b) for forward edges
        # perm1: (a,b) -> (a, b+1 mod 3) for within-layer
        # perm2: (a,b) -> (a-1 mod 3, b) for backward edges
        
        # Actually, let's use a more careful construction
        # The key is to avoid short cycles
        
        # Use a product structure:
        # Layer i has 9 vertices: (a,b) for a,b in {0,1,2}
        # Forward edges from layer i to layer i+1
        # Backward edges from layer i to layer i-1
        # Within-layer edges
        
        # For girth 16 with 192 vertices, L=16 might work
        # Let's try L values and certify
        
        for L in [8, 12, 16, 24, 32]:
            if time.time() - t0 > deadline - 10:
                break
                
            n = 12 * L
            edges = []
            eset = set()
            
            # Vertex encoding: (i, a, b) -> i * 9 + a * 3 + b
            def vid(i, a, b):
                return i * 9 + a * 3 + b
            
            # Forward edges: (i, a, b) -> ((i+1)%L, (a+1)%3, b)
            # Backward edges: (i, a, b) -> ((i-1)%L, (a+2)%3, b)
            # Within edges: (i, a, b) -> (i, a, (b+1)%3)
            
            for i in range(L):
                for a in range(3):
                    for b in range(3):
                        v = vid(i, a, b)
                        
                        # Forward edge
                        fi = (i + 1) % L
                        fa = (a + 1) % 3
                        fb = b
                        fv = vid(fi, fa, fb)
                        key = (min(v, fv), max(v, fv))
                        if key not in eset:
                            eset.add(key)
                            edges.append(key)
                        
                        # Backward edge
                        bi = (i - 1) % L
                        ba = (a + 2) % 3
                        bb = b
                        bv = vid(bi, ba, bb)
                        key = (min(v, bv), max(v, bv))
                        if key not in eset:
                            eset.add(key)
                            edges.append(key)
                        
                        # Within edge
                        wa = a
                        wb = (b + 1) % 3
                        wv = vid(i, wa, wb)
                        key = (min(v, wv), max(v, wv))
                        if key not in eset:
                            eset.add(key)
                            edges.append(key)
            
            # Check degree
            deg = [0] * n
            ok = True
            for u, v in edges:
                if u == v:
                    ok = False
                    break
                deg[u] += 1
                deg[v] += 1
            if not ok:
                continue
            if any(x != 3 for x in deg):
                continue
            
            if time.time() - t0 > deadline - 5:
                break
                
            if certify_girth(n, edges, g):
                return n, edges
        
        return None, None

    def luw_construct(k, g, d):
        q = k
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

    best_n, best_edges = None, None

    if k == 3 and g == 16:
        # Try Bosak construction first (much smaller graphs)
        n_b, edges_b = bosak_construct(k, g)
        if n_b is not None:
            best_n, best_edges = n_b, edges_b
        
        # If Bosak didn't work or deadline allows, try LUW as fallback
        if best_n is None and time.time() - t0 < deadline - 30:
            d_candidates = [8, 9, 10]
            for d in d_candidates:
                if time.time() - t0 > deadline - 10:
                    break
                try:
                    n, edges = luw_construct(k, g, d)
                except Exception:
                    continue
                deg = [0] * n
                ok = True
                for u, v in edges:
                    if u == v:
                        ok = False
                        break
                    deg[u] += 1
                    deg[v] += 1
                if not ok:
                    continue
                if any(x != k for x in deg):
                    continue
                if time.time() - t0 > deadline - 10:
                    break
                if certify_girth(n, edges, g):
                    if best_n is None or n < best_n:
                        best_n, best_edges = n, edges
                    break

    if best_n is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_edges = luw_construct(k, g, d)

    return best_n, best_edges
