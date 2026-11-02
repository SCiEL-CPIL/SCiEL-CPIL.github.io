"""Construct a k-regular graph of girth at least g, minimizing vertex count."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def time_left():
        return deadline - time.time()

    # Moore bound for lower bound on n
    if g % 2 == 0:
        moore = k + k * (k - 1) ** (g // 2 - 1)
    else:
        if k == 2:
            moore = g
        else:
            h = (g - 1) // 2
            moore = 1 + k * ((k - 1) ** h - 1) // (k - 2) if k > 2 else g

    best_n = None
    best_edges = None

    def verify_girth(n, edges, min_girth):
        """Verify no cycle shorter than min_girth using non-backtracking BFS from each vertex."""
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Check simple: no self-loops, no duplicate edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        # Non-backtracking BFS: state is (node, prev_node)
        # dist[(u, v)] = length of shortest non-backtracking walk from start to v via u
        # A cycle of length L exists iff for some edge (u,v), dist[(u,v)] + dist[(v,u)] + 1 < min_girth
        # Actually simpler: BFS in the directed line graph, looking for cycles
        for start in range(n):
            # dist[prev][curr] = distance
            dist = {}
            q = deque()
            for nb in adj[start]:
                dist[(start, nb)] = 1
                q.append((start, nb))
            while q:
                prev, curr = q.popleft()
                d = dist[(prev, curr)]
                for nb in adj[curr]:
                    if nb == prev:
                        continue  # non-backtracking
                    if (curr, nb) in dist:
                        # Found a cycle: d + 1 (the edge curr->nb) closes it
                        # The cycle length is d + 1
                        cycle_len = d + 1
                        if cycle_len < min_girth:
                            return False
                    else:
                        if d + 1 < min_girth:
                            dist[(curr, nb)] = d + 1
                            q.append((curr, nb))
        return True

    def try_explicit_construction(d_target):
        """
        The Lazebnik-Ustimenko-Woldar construction.
        We try to use a specific depth d_target.
        """
        q = k # The seed uses q=k. For k=5, q=5.
        
        # Field operations
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return x * y % q

        # Rule definition
        # The seed defines rules up to d.
        # We need to ensure rules are defined for all c in 2..d_target
        rule = {}
        if d_target >= 2:
            rule[2] = (1, 1)
        if d_target >= 3:
            rule[3] = (2, 1)
        if d_target >= 4:
            rule[4] = (1, 2)
        for c in range(5, d_target + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                # x is a tuple of length d_target + 1?
                # In seed: y = [None, first] + [0] * (d - 1)
                # This creates a list of length 1 + 1 + d - 1 = d + 1
                # Indices: 0 is None, 1 is first, 2..d are computed
                y = [None, first] + [0] * (d_target - 1)
                for c in range(2, d_target + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_target)
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

    # Strategy: Try smaller depths first to get smaller n.
    # n = 2 * k^d for the L-U-W construction.
    # We need girth >= g. Try d in increasing order; first valid one is optimal.
    
    # Minimum d needed: girth of L-U-W with depth d is at least 2*d or 2*d+1 depending on structure.
    # For safety, start from d that could potentially give girth >= g.
    # Empirically, d >= (g-1)//2 often suffices, but we verify.
    
    max_d = 10  # Safety cap; 2*5^10 is huge, won't be constructed in time
    for d in range(3, max_d + 1):
        if time_left() < 15:
            break
            
        # Estimate n = 2 * k^d; if too large, skip verification (construction itself may be slow)
        est_n = 2 * (k ** d)
        if est_n > 50000:
            # Construction may be too slow or memory-heavy; but let's try if time permits
            if time_left() < 30:
                break
            
        n_try, edges_try = try_explicit_construction(d)
        
        if verify_girth(n_try, edges_try, g):
            if best_n is None or n_try < best_n:
                best_n = n_try
                best_edges = edges_try
            # Since n increases with d, first valid d gives smallest n
            break

    if best_n is None:
        # Fallback: use a larger d that is guaranteed to have sufficient girth
        d_seed = max(3, g - 2)
        n_exp, edges_exp = try_explicit_construction(d_seed)
        return n_exp, edges_exp

    return best_n, best_edges
# EVOLVE-BLOCK-END
