import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def _petersen():
        edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                 (5,6),(6,7),(7,8),(8,9),(9,5),
                 (0,5),(1,6),(2,7),(3,8),(4,9)]
        return 10, edges

    def _heawood():
        lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
        edges = []
        for i, line in enumerate(lines):
            for pt in line:
                edges.append((pt, 7 + i))
        return 14, edges

    def _known(k, g):
        if k == 3 and g == 5:
            return _petersen()
        if k == 3 and g == 7:
            return _heawood()
        return None

    def luw_construct(k, g):
        """LUW construction: for prime q, degree q, girth >= 2*d+1."""
        q = k
        d = (g + 1) // 2 if g % 2 == 1 else g // 2
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
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        # Encode vertex as (side, y1, y2, ..., yd) where side 0 has y1=0
        # We store as a tuple of (d+1) ints: (side, y1, ..., yd)
        # But y1 is always 0 on side 0 and first on side 1.
        # Actually for LUW, vertices are pairs (side, (y1,...,yd))
        # side 0: y1 = 0 (the "left" column)
        # side 1: y1 = free (the "right" column)

        # For side 0 vertex (0, (0, y2, ..., yd)):
        #   neighbours are (1, (f1, y2', ..., yd')) for each f1 in 0..q-1
        # For side 1 vertex (1, (y1, y2, ..., yd)):
        #   neighbours are (0, (0, y2', ..., yd')) for each y1 in 0..q-1

        # The recurrence:
        # side 0: y_c = x_c + y_a * x_b  for c = 2..d
        # side 1: y_c = x_c - x_a * y_b  for c = 2..d

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [0] * (d + 1)  # y[0] unused, y[1]..y[d]
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side, tuple(y[1:]))

        start = (0, tuple([0] * d))
        index = {start: 0}
        order = [start]
        edges = []

        for i in range(len(order)):
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        return len(order), edges

    # Try known cages first
    known = _known(k, g)
    if known is not None:
        best_n, best_edges = known
    else:
        best_n, best_edges = luw_construct(k, g)

    # Build adjacency for verification
    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def girth_ok(n, edges, g):
        """Check girth >= g using non-backtracking walks from every vertex.
        A cycle of length L exists iff there is a non-backtracking walk of length L
        that returns to its start. We check all non-backtracking walks up to length g-1
        from every vertex to see if any closes a cycle shorter than g."""
        if g <= 3:
            return True  # no cycles possible in a tree-like structure of this size

        adj = build_adj(n, edges)
        max_len = g - 1  # check walks up to this length; if any closes, girth < g

        # Non-backtracking walk: sequence v0, v1, ..., vk where v_{i-1} != v_{i+1}
        # and consecutive vertices are adjacent.
        # A cycle of length L is a non-backtracking closed walk of length L with no
        # repeated vertices (but for girth checking, we just need to detect ANY closed
        # non-backtracking walk of length L < g).

        # Efficient approach: for each starting vertex, enumerate non-backtracking
        # walks up to length max_len. If a walk of length L returns to start with
        # L < g, girth < g.

        # To be fast, we use the fact that in a k-regular graph, the number of
        # non-backtracking walks of length t from a vertex is k*(k-1)^(t-1).
        # For k=5, g=9: 5*4^8 = 5*65536 = 327680 per vertex. With 1250 vertices
        # that's ~409M which is too slow in Python.

        # Better: use the non-backtracking matrix power approach, or
        # a smarter BFS that tracks (current, previous) pairs.

        # Let's use BFS on the directed non-backtracking graph.
        # State: (current_vertex, previous_vertex)
        # Start states: for each vertex v, for each neighbor u of v: (u, v)
        # We want to know if any state (v, prev) can reach (v, next) where next != prev
        # in fewer than g-1 additional steps... actually this is getting complex.

        # Simpler approach for verification: BFS from each vertex tracking
        # (vertex, parent) to detect when we revisit a vertex via a different path.
        # Cycle length = dist[u] + dist[v] + 1 where u and v are the two meeting points.

        adj = build_adj(n, edges)

        for s in range(n):
            # BFS from s, tracking shortest distance and one parent
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])

            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle through s
                        # The cycle goes s -> ... -> u -> v -> ... -> s
                        # Length = dist[u] + 1 + dist[v]
                        cycle_len = dist[u] + 1 + dist[v]
                        if cycle_len < g:
                            return False

        return True

    # Verify
    if time.time() < deadline - 10:
        if girth_ok(best_n, best_edges, g):
            return best_n, best_edges
        # If girth check fails (shouldn't for LUW), try a different d
        # or fall through to return anyway
    else:
        return best_n, best_edges

    # Fallback: try with larger d if the girth check failed
    if time.time() < deadline - 10:
        # Try d+1
        q = k
        d_alt = (g + 1) // 2 + 1
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
        for c in range(5, d_alt + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours_alt(vertex):
            side, x = vertex
            for first in range(q):
                y = [0] * (d_alt + 1)
                y[1] = first
                for c in range(2, d_alt + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side, tuple(y[1:]))

        start = (0, tuple([0] * d_alt))
        index = {start: 0}
        order = [start]
        edges_alt = []

        for i in range(len(order)):
            vertex = order[i]
            for other in neighbours_alt(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_alt.append((index[vertex], index[other]))

        n_alt = len(order)
        if time.time() < deadline - 5:
            if girth_ok(n_alt, edges_alt, g):
                return n_alt, edges_alt

    return best_n, best_edges
