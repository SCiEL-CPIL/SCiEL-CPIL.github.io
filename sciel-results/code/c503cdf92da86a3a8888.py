"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import deque

def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    # d is the parameter; girth of CD(d,q) is d+5 for odd d, d+6 for even d
    # We want girth >= g. For odd d, girth = d+5 >= g => d >= g-5.
    # Choose d so that d+5 >= g and d is odd.
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)

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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _verify_girth(n, edges, k, g):
    """Verify every vertex has degree k, graph is simple, and girth >= g via BFS from each vertex."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u >= n or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Check girth: BFS from each vertex, track parent to avoid backtracking
    for start in range(n):
        # BFS layers; dist[v] = distance, parent[v] = parent in BFS tree
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        found_short = False
        while queue:
            u = queue.popleft()
            if dist[u] >= g:
                break
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < g:
                        queue.append(v)
                else:
                    # found a cycle of length dist[u] + dist[v] + 1
                    if dist[u] + dist[v] + 1 < g:
                        found_short = True
                        break
            if found_short:
                break
        if found_short:
            return False
    return True


def _moore_bound(k, g):
    """Lower bound on number of vertices for a k-regular graph of girth g."""
    if g % 2 == 1:
        # Odd girth: 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        r = (g - 3) // 2
        if k == 1:
            return 1 + 1
        total = 1 + k
        for i in range(1, r + 1):
            total += k * (k - 1) ** i
        return total
    else:
        # Even girth: 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
        r = g // 2 - 1
        if k == 1:
            return 2
        total = 0
        for i in range(r + 1):
            total += (k - 1) ** i
        return 2 * total


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 140  # safety margin before 150s kill

    # Start with the known explicit construction
    n_luw, edges_luw = _construct_luw(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # Verify the LUW construction
    if not _verify_girth(n_luw, edges_luw, k, g):
        # Should not happen, but fallback
        pass

    # Try to improve via local search: attempt to find smaller graphs
    # For small k and g, we can try random regular graph generation and check girth
    # This is only feasible for small instances

    if k <= 3 and g <= 8:
        # Small enough to try random search
        random.seed(42)
        mb = _moore_bound(k, g)
        # Try a few random k-regular graphs of increasing size
        for trial_n in range(mb, min(mb + 50, n_luw)):
            if time.time() > deadline:
                break
            # Generate random k-regular graph
            for attempt in range(20):
                if time.time() > deadline:
                    break
                try:
                    edges = _random_regular_graph(trial_n, k)
                    if edges is not None:
                        if _verify_girth(trial_n, edges, k, g):
                            best_n = trial_n
                            best_edges = edges
                            break
                except Exception:
                    pass
            if best_n == trial_n:
                break

    return best_n, best_edges


def _random_regular_graph(n, k):
    """Generate a random k-regular simple graph on n vertices, or None if impossible."""
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    edges = []
    edge_set = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None
        edge_set.add(key)
        edges.append((u, v))
    # Verify degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != k for d in deg):
        return None
    return edges
# EVOLVE-BLOCK-END
