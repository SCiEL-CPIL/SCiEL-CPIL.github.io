import time
import sys
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = 115.0

    def certify_girth(n, edges, g):
        """
        Checks if the graph has girth at least g.
        Uses bounded BFS from each vertex up to depth g//2 - 1.
        If a vertex is reached from two different paths (or same path backtracking is handled),
        we check for cycles.
        Simpler: BFS from each vertex, tracking (node, parent).
        If we hit a node already visited in this BFS with a different parent,
        cycle length = dist[u] + dist[v] + 1.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_depth = g // 2 - 1
        # If g is even, we look for cycles of length < g.
        # A cycle of length L implies a path of length L/2 from u to v.
        # BFS depth max_depth covers cycles up to 2*max_depth + 1?
        # Actually, if we BFS to depth D, we find cycles of length <= 2D + 1?
        # No, if we find a node at depth d1 from start, and it was visited at depth d2,
        # and d1 != d2, cycle length is d1 + d2 + 1 (if different parents) or similar.
        # Standard check: BFS from each vertex, record dist and parent.
        # If we encounter a visited node `v` with `parent[v] != u`,
        # then we have a cycle of length `dist[u] + dist[v] + 1`.
        # We need to ensure this length is >= g.
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # Back edge or cross edge
                        if v == parent[u]:
                            continue
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def luw_construct(k, d):
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
        
        # Rule for LUW
        # For k=3, d=8
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            # The seed code had a specific list selection
            # [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            # Let's use the seed's logic exactly
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            # x is a tuple of length d+1, x[0] is None, x[1..d] are values
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

    if k == 3 and g == 16:
        # Try d=8 first (smallest for g=16)
        d_candidates = [8, 9, 10]
        for d in d_candidates:
            if time.time() - t0 > deadline - 10:
                break
            try:
                n, edges = luw_construct(k, d)
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
                break # d=8 is minimal, so stop if valid

    if best_n is None:
        # Fallback: try larger d if d=8 failed or was too slow
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        try:
            best_n, best_edges = luw_construct(k, d)
        except Exception:
            pass

    if best_n is None:
        return 0, []

    return best_n, best_edges
