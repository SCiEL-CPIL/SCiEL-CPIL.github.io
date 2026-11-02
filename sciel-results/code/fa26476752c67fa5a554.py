import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construction():
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

    def has_short_cycle_fast(adj, g):
        """Fast girth check: return True if any cycle of length < g exists.
        Uses non-backtracking BFS with early termination and compact state encoding."""
        n = len(adj)
        # Precompute: for each directed edge (u,v), the next states are (v,w) for w in adj[v], w != u
        # State: (v, p) encoded as v * n + p (only valid when p in adj[v])
        # For girth check, we need to find closed non-backtracking walks of length < g
        
        # For each start vertex s and first neighbor a, BFS in non-backtracking space
        # State (v, p): at v, came from p. Distance = number of edges so far.
        # Start: (a, s) at dist 1. Goal: reach (s, b) for b in adj[s], b != a, at dist < g.
        
        for s in range(n):
            neighbors_s = adj[s]
            for a in neighbors_s:
                # BFS from (a, s)
                # Use dict: state -> dist
                dist = {(a, s): 1}
                frontier = [(a, s)]
                for depth in range(1, g - 1):
                    next_frontier = []
                    for v, p in frontier:
                        for w in adj[v]:
                            if w == p:
                                continue
                            if w == s and v != a:
                                # Found cycle of length depth + 1 < g
                                return True
                            state = (w, v)
                            if state not in dist:
                                dist[state] = depth + 1
                                next_frontier.append(state)
                    frontier = next_frontier
                    if not frontier:
                        break
        return False

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def random_k_regular(n, k, seed=None):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if n * k % 2 != 0:
            return None
        rng = random.Random(seed)
        half_edges = list(range(n)) * k
        rng.shuffle(half_edges)
        edges = set()
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                return None
            key = (min(u, v), max(u, v))
            if key in edges:
                return None
            edges.add(key)
            adj[u].append(v)
            adj[v].append(u)
        return sorted([list(e) for e in edges])

    # First, get the LUW construction as baseline
    n_luw, edges_luw = luw_construction()
    best_n, best_edges = n_luw, edges_luw

    # Moore bound for (4,10): 2 * sum_{i=0}^{4} 3^i = 242
    moore_bound = 2 * sum((k-1)**i for i in range(g // 2))
    if moore_bound % 2 != 0:
        moore_bound += 1

    # Try candidate sizes from Moore bound up to best_n - 2
    # Focus on sizes where random graphs might work
    # For 4-regular girth 10, try n = 244, 246, ..., up to best_n
    candidates = list(range(moore_bound, best_n, 2))
    
    # Prioritize larger n (closer to known best) as they're more likely to exist
    # but also try smaller ones
    # Actually, let's try in order from smallest to find the minimum
    # But limit total work
    
    time_per_check_estimate = 0.5  # rough estimate per girth check
    max_checks = int((deadline - time.time() - 10) / max(0.1, time_per_check_estimate))
    
    # Try each candidate size with a few random graphs
    for n_try in candidates:
        if time.time() > deadline - 10:
            break
        # Try up to 50 random graphs for this size
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            edges = random_k_regular(n_try, k, seed=attempt * 1000 + n_try)
            if edges is None:
                continue
            adj = build_adj(edges, n_try)
            if not has_short_cycle_fast(adj, g):
                best_n, best_edges = n_try, edges
                break
        if best_n == n_try:
            break  # Found a valid graph at this size, try smaller

    return best_n, best_edges
