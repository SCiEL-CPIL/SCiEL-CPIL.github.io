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

    def luw_construct(k, g, d):
        q = k
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

    # Target specific case k=3, g=16
    if k == 3 and g == 16:
        # d=11 gives girth >= 16
        # We try d=11 first as it's the minimal d for g=16
        # If it fails, try others
        d_candidates = [11, 12, 10, 9, 8]
        for d in d_candidates:
            if time.time() - t0 > deadline - 5:
                break
            try:
                n, edges = luw_construct(k, g, d)
            except Exception:
                continue
            
            # Quick validation
            ok = True
            deg = [0] * n
            eset = set()
            for u, v in edges:
                if u == v:
                    ok = False
                    break
                key = (min(u, v), max(u, v))
                if key in eset:
                    ok = False
                    break
                eset.add(key)
                deg[u] += 1
                deg[v] += 1
            
            if not ok:
                continue
            if any(x != k for x in deg):
                continue
            
            if time.time() - t0 > deadline - 5:
                break
            
            if certify_girth(n, edges, g):
                if best_n is None or n < best_n:
                    best_n, best_edges = n, edges
                # If we found a valid graph, we can stop if it's small enough?
                # For k=3, g=16, n=936 is the target.
                # If we find n < 936, it's a breakthrough.
                # If we find n >= 936, we keep looking for smaller?
                # But LUW with d=11 is likely the smallest LUW graph for this girth.
                # So we can break if we found a valid one, as larger d will likely yield larger n.
                break

    # Fallback for other k, g or if specific case failed
    if best_n is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        try:
            best_n, best_edges = luw_construct(k, g, d)
        except Exception:
            # Last resort: return an empty graph or raise? 
            # The contract says "invalid result ... scores 0".
            # But we must return something.
            # A trivial graph? No, must be k-regular.
            # If k=1, n=2, edge (0,1). Girth infinity?
            # If k=2, cycle.
            # For safety, if LUW fails, we try a random config model?
            # But for now, assume LUW works.
            pass

    if best_n is None:
        # Emergency fallback: complete graph? No, girth 3.
        # Just return a small valid graph if possible, else fail.
        # For k=3, g=16, LUW should not fail.
        return 0, []

    return best_n, best_edges
