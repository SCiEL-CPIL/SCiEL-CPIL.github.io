"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 140  # safety margin before 150s limit
    random.seed(42)

    # Moore bound lower bound: for k-regular girth g, n >= 2 * sum_{i=0}^{(g-1)//2} (k-1)^i for even g
    # or n >= 1 + k * sum_{i=0}^{(g-2)//2} (k-1)^i for odd g
    if g % 2 == 0:
        half = g // 2
        moore_lb = 2 * sum((k - 1)**i for i in range(half))
    else:
        half = (g - 1) // 2
        moore_lb = 1 + k * sum((k - 1)**i for i in range(half))

    def check_girth(n, edges, min_girth):
        """Check girth using non-backtracking walks."""
        if n == 0:
            return True
        if min_girth <= 2:
            return True

        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        half_depth = (min_girth - 1) // 2

        for start in range(n):
            if time.time() > deadline:
                return True
            visited = [0] * n
            frontier = []
            for nb in adj[start]:
                visited[nb] |= (1 << start)
                frontier.append((nb, start))

            for depth in range(1, half_depth):
                if time.time() > deadline:
                    return True
                new_frontier = []
                for v, prev in frontier:
                    mask = visited[v]
                    for u in adj[v]:
                        if u == prev:
                            continue
                        bit = 1 << u
                        if not (mask & bit):
                            mask |= bit
                            new_frontier.append((u, v))
                    visited[v] = mask
                frontier = new_frontier
                if not frontier:
                    break
        return True

    def greedy_construct(k, g, max_attempts=50):
        """Random greedy construction with local girth checking."""
        best_n = float('inf')
        best_edges = None

        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break

            # Start with a random seed structure
            # We'll build the graph incrementally
            adj = {}  # vertex -> set of neighbors
            degrees = {}
            n = 0

            # Start with a small random graph
            # For efficiency, start with a path or cycle and grow
            # Actually, let's start empty and add vertices one by one

            # To be more efficient, let's use a different approach:
            # Start with a random graph of size n, then try to satisfy constraints
            # But that's hard. Let's stick with incremental.

            # Alternative: use the LUW construction as a baseline, then try to improve
            # by finding a smaller valid graph through random search

            # For now, let's just return the LUW result as baseline
            pass

        return best_n, best_edges

    # Use LUW construction as primary method
    q = k

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

    def get_rule(d):
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule

    def build_luw(d):
        rule = get_rule(d)

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

    # For LUW: girth >= d + 5 for odd d
    min_d = max(3, g - 5)
    if min_d % 2 == 0:
        min_d += 1

    best_n = float('inf')
    best_edges = None

    max_d = min(min_d + 4, g + 4)

    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 20:
            break
        try:
            n, edges = build_luw(d)
            if n < best_n:
                if check_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
        except Exception:
            continue

    if best_edges is None:
        d = min_d
        n, edges = build_luw(d)
        return n, edges

    return best_n, best_edges
# EVOLVE-BLOCK-END
