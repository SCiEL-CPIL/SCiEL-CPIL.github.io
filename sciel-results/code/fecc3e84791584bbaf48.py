import sys
import time
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0

    if k == 5 and g == 9:
        # Strategy: LUW(q=5, d=5) gives ~3125 vertices with girth >= 10.
        # Add 2 perfect matchings between points and lines at distance >= 8
        # to get a 5-regular graph on ~1562 vertices.
        try:
            n, edges = construct_luw_q5_d5_plus_2matchings()
            if n is not None and n > 0:
                # Verify degree and girth
                if verify_girth(n, edges, k, g):
                    return n, edges
        except Exception:
            pass

        # Fallback: plain LUW(q=5, d=5) without matchings
        try:
            n, edges = construct_luw_plain(5, 5)
            if n is not None and n > 0:
                if verify_girth(n, edges, k, g):
                    return n, edges
        except Exception:
            pass

    # Generic fallback
    return construct_default(k, g)


def construct_luw_plain(q, d):
    """Build the LUW CD(d,q) graph and return (n, edges)."""
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
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
            y = [0, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    adj = [[] for _ in range(20000)]

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                u_idx = index[vertex]
                v_idx = index[other]
                edges.append((u_idx, v_idx))
                adj[u_idx].append(v_idx)
                adj[v_idx].append(u_idx)

    return len(order), edges, adj, order


def construct_luw_q5_d5_plus_2matchings():
    """
    Build LUW(5,5) graph (~3125 vertices, 3-regular bipartite),
    then add 2 perfect matchings between points and lines where
    distance >= 8, yielding a 5-regular graph on ~1562 vertices.
    """
    q = 5
    d = 5

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
            y = [0, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    adj = [[] for _ in range(20000)]

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                u_idx = index[vertex]
                v_idx = index[other]
                edges.append((u_idx, v_idx))
                adj[u_idx].append(v_idx)
                adj[v_idx].append(u_idx)

    n = len(order)
    points = [i for i in range(n) if order[i][0] == 0]
    lines = [i for i in range(n) if order[i][0] == 1]
    n_p = len(points)
    n_l = len(lines)

    if n_p != n_l:
        return None, None

    # Compute distances from each point to all lines using BFS.
    # We only need point-to-line distances, so n_p BFS runs instead of n.
    # Store as a list of arrays: point_dist[p_idx][v] = distance
    # But to save memory, we only need to check dist >= 8, so we can
    # do BFS up to depth 7 and mark which lines are reached.

    # For each point, BFS up to depth 7, collect lines at distance >= 8
    # Actually, we need lines NOT reached within 7 steps (i.e., dist >= 8)
    # Or lines reached at exactly dist >= 8.
    # Since the graph has girth >= 10, no line is within distance 3 of a point
    # (that would create a 6-cycle). Lines at distance 4,5,6,7 are "close".
    # We want lines at distance >= 8.

    # Build match adjacency: for each point, which lines are at distance >= 8
    match_adj = [[] for _ in range(n_p)]

    for pi, p in enumerate(points):
        # BFS from p up to depth 7
        visited = np.zeros(n, dtype=np.bool_)
        visited[p] = True
        queue = deque([p])
        depth = 0
        while queue and depth < 7:
            next_queue = deque()
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        next_queue.append(v)
            queue = next_queue
            depth += 1

        # Lines at distance >= 8 are those not visited
        for li, l in enumerate(lines):
            if not visited[l]:
                match_adj[pi].append(li)

    # Check feasibility
    for i in range(n_p):
        if len(match_adj[i]) < 2:
            return None, None

    # Find two disjoint perfect matchings using Hopcroft-Karp
    def hopcroft_karp(adj_list, n_left, n_right, forbidden=None):
        """Find maximum matching. forbidden is a set of (u,v) pairs to exclude."""
        if forbidden is None:
            forbidden = set()

        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist_hk = [0] * n_left

        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist_hk[u] = 0
                    queue.append(u)
                else:
                    dist_hk[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj_list[u]:
                    if (u, v) in forbidden:
                        continue
                    if pair_v[v] == -1:
                        found = True
                    elif dist_hk[pair_v[v]] == float('inf'):
                        dist_hk[pair_v[v]] = dist_hk[u] + 1
                        queue.append(pair_v[v])
            return found

        def dfs(u):
            for v in adj_list[u]:
                if (u, v) in forbidden:
                    continue
                if pair_v[v] == -1 or (dist_hk[pair_v[v]] == dist_hk[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist_hk[u] = float('inf')
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching, pair_u, pair_v

    # First matching
    m1, pair_u1, pair_v1 = hopcroft_karp(match_adj, n_p, n_l)
    if m1 < n_p:
        return None, None

    new_edges = list(edges)
    forbidden = set()
    for u in range(n_p):
        v = pair_u1[u]
        if v != -1:
            p = points[u]
            l = lines[v]
            new_edges.append((p, l))
            forbidden.add((u, v))

    # Second matching (disjoint from first)
    m2, pair_u2, pair_v2 = hopcroft_karp(match_adj, n_p, n_l, forbidden)
    if m2 < n_p:
        return None, None

    for u in range(n_p):
        v = pair_u2[u]
        if v != -1:
            p = points[u]
            l = lines[v]
            new_edges.append((p, l))

    return n, new_edges


def construct_default(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

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
            y = [0, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
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


def verify_girth(n, edges, k, g):
    """Verify degree and girth using BFS from each edge endpoint."""
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        adj[u].append(v)
        adj[v].append(u)
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)

    for i in range(n):
        if len(adj[i]) != k:
            return False

    # Girth check: for each edge (u,v), shortest path from u to v avoiding (u,v)
    # If that path length + 1 < g, girth is violated.
    for u, v in edges:
        dist = [-1] * n
        dist[u] = 0
        queue = deque([u])
        found = False
        while queue:
            cur = queue.popleft()
            if cur == v:
                if dist[v] + 1 < g:
                    return False
                found = True
                break
            for nb in adj[cur]:
                if (cur == u and nb == v) or (cur == v and nb == u):
                    continue
                if dist[nb] == -1:
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        if not found:
            return False

    return True
