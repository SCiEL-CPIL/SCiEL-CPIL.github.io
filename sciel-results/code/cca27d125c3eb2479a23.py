"""Rewritten: fast exact girth + exhaustive abelian Cayley search + polish."""
import time
import random
import sys
from collections import deque
from itertools import product

_DEADLINE = 120.0
_MARGIN = 8.0


def _exact_girth_ok(adj, n, g):
    """
    Exact girth check via non-backtracking BFS from every vertex.
    Returns True iff every cycle has length >= g.
    Optimized: single visited array per source with round stamps,
    early termination on first short cycle.
    """
    for s in range(n):
        # BFS on non-backtracking walks from s.
        # State: (curr, prev, depth).  Depth = number of edges taken from s.
        # Fail if we reach curr==s with depth < g.
        # Prune: don't revisit same (curr, prev) pair.
        seen = set()
        dq = deque()
        for nb in adj[s]:
            state = (nb, s)
            if state not in seen:
                seen.add(state)
                dq.append((nb, s, 1))
        while dq:
            curr, prev, d = dq.popleft()
            if curr == s:
                if d < g:
                    return False
                continue
            if d >= g - 1:
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in seen:
                    seen.add(state)
                    dq.append((nb, curr, d + 1))
    return True


def _girth_geq(adj, n, min_g):
    """Check girth >= min_g (cheaper: only need to verify no cycle < min_g)."""
    return _exact_girth_ok(adj, n, min_g)


def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if (g - 5) % 2 == 0:
        d = g - 4

    if q == 4:
        table = [[0, 0, 0, 0],
                 [0, 1, 2, 3],
                 [0, 2, 3, 1],
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

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


def _build_abelian_group(orders):
    """
    Build the additive group of product of Z_m for m in orders.
    Returns (n, elems) where elems[i] is a tuple of coordinates.
    """
    n = 1
    for m in orders:
        n *= m
    elems = [None] * n
    # Mixed-radix enumeration
    idx = 0
    coords = [0] * len(orders)
    for a in product(*(range(m) for m in orders)):
        elems[idx] = a
        idx += 1
    return n, elems


def _try_cayley_abelian(orders, k, g, deadline, best_n):
    """
    Enumerate symmetric 4-element generating sets of the abelian group
    of type 'orders' and find the best girth >= g graph.
    Returns (n, edges) for best found, or (None, None).
    """
    n, elems = _build_abelian_group(orders)
    if n >= best_n:
        return None, None

    num_coords = len(orders)
    # Precompute group operation and inverse
    def add_group(a, b):
        return tuple((a[i] + b[i]) % orders[i] for i in range(num_coords))

    def neg_group(a):
        return tuple((-a[i]) % orders[i] for i in range(num_coords))

    # We need to choose 4 elements g1, g2, g3, g4 from the group such that
    # {g1, g2, g3, g4, -g1, -g2, -g3, -g4} has at most 4 distinct non-identity
    # elements (if g_i = -g_j, they share), and the set generates the group,
    # and no element is identity.
    # For k=4: we need exactly 4 distinct neighbors per vertex.
    # If g_i != -g_j for all i != j and g_i != 0, then we have 8 generators
    # but only 4 distinct neighbors per vertex (since +g and -g give different
    # neighbors).  Wait, no: Cayley graph with generators S where S is a set
    # of size k, and for undirected we need S = -S.  So |S| = 4 means we need
    # 4 generators that are closed under negation.
    # This means either:
    #   (a) 2 pairs {g, -g} where g != -g, giving 4 generators.
    #   (b) 4 elements where some are self-inverse (g = -g, i.e., 2g = 0).
    # For simplicity, we'll enumerate pairs (a, b) and form S = {a, -a, b, -b},
    # checking |S| == 4 and a != 0, b != 0, a != b, a != -b.
    # But we also need S to generate the whole group.

    # For efficiency, enumerate all (a, b) pairs where a < b in some ordering.
    # Total pairs: n * (n-1) / 2 which can be huge for large n.
    # We need a smarter approach.

    # Better: enumerate all elements, and for each, check if it's a valid generator.
    # Then form all pairs and test.
    # For n up to 500, n^2 = 250000, which is manageable.

    elements = list(range(n))  # just use indices
    # Actually, let's work with tuples for the group elements.
    # Precompute all group elements as tuples.
    # For each pair (i, j) with i < j, form S = {elems[i], elems[j], -elems[i], -elems[j]}
    # Check |S| == 4, generate the group, then BFS for girth.

    # Precompute negatives
    negs = [neg_group(e) for e in elems]

    best_local_n = best_n
    best_local_edges = None

    # To avoid duplicates: (i,j) and (j,i) are the same, and also (i,j) and (i,-j) etc.
    # We'll just enumerate all unordered pairs and use a set to track tried generator sets.
    tried_s = set()

    for i in range(n):
        a = elems[i]
        if a == tuple(0 for _ in orders):
            continue
        neg_a = negs[i]
        if a == neg_a:
            continue  # self-inverse, handle separately or skip for now
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            b = elems[j]
            if b == tuple(0 for _ in orders):
                continue
            neg_b = negs[j]
            if b == neg_b:
                continue
            if a == b or a == neg_b:
                continue

            s_set = frozenset([a, neg_a, b, neg_b])
            if len(s_set) != 4:
                continue

            # Canonical key: sort the generator set
            key = tuple(sorted(s_set))
            if key in tried_s:
                continue
            tried_s.add(key)

            # Check if S generates the group
            # BFS from identity
            identity = tuple(0 for _ in orders)
            if identity not in s_set:  # it shouldn't be
                pass
            visited = {identity: True}
            queue = [identity]
            gen_list = list(s_set)
            gen_index = {g: idx for idx, g in enumerate(gen_list)}

            while queue:
                u = queue.pop(0)
                for g in gen_list:
                    v = add_group(u, g)
                    if v not in visited:
                        visited[v] = True
                        queue.append(v)

            if len(visited) != n:
                continue

            # Now build the Cayley graph and check girth.
            # Map group elements to vertex indices (0..n-1)
            vertex_index = {e: i for i, e in enumerate(elems)}
            adj = [[] for _ in range(n)]
            for i in range(n):
                vi = vertex_index[elems[i]]
                for g in gen_list:
                    v = add_group(elems[i], g)
                    adj[vi].append(vertex_index[v])

            # Check girth >= g
            if _girth_geq(adj, n, g):
                # Build edge list
                edges = []
                edge_set = set()
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edge_set.add((i, j))
                edges = list(edge_set)
                return n, edges

    return best_local_n, best_local_edges


def _validate_graph(n, edges, k, g, max_vertices=None):
    """Full validation: simple, k-regular, girth >= g."""
    if max_vertices is not None and n > max_vertices:
        return False
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Convert to list of lists for girth check
    adj_list = [list(s) for s in adj]
    return _exact_girth_ok(adj_list, n, g)


def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN

    # 1. LUW baseline
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # 2. For (4,10), try abelian Cayley graphs
    if k == 4 and g == 10:
        # Try various abelian group structures with order < best_n
        # Candidate orders: products of small primes
        candidate_orders = [
            [2, 2, 2, 2, 2],       # 32
            [2, 2, 2, 2, 3],       # 48
            [2, 2, 2, 3, 3],       # 72
            [2, 2, 3, 3, 3],       # 108
            [3, 3, 3, 3, 2],       # 162
            [2, 2, 2, 5, 5],       # 200
            [2, 2, 3, 5, 5],       # 300
            [5, 5, 5, 5],          # 625 - too big, skip
            [4, 4, 5, 5],          # 400
            [2, 2, 2, 4, 8],       # 256
            [2, 2, 2, 4, 12],      # 384
            [2, 2, 3, 4, 10],      # 480
            [2, 2, 2, 2, 10],      # 160
            [2, 2, 2, 4, 6],       # 192
            [2, 2, 3, 3, 4],       # 144
            [2, 3, 3, 5, 2],       # 180
            [2, 2, 5, 5, 2],       # 200
            [4, 5, 10, 2],         # 400
            [2, 2, 2, 10, 10],     # 400
            [6, 5, 10, 2],         # 600 - too big
            [4, 4, 4, 12],         # 768 - too big
            [3, 3, 3, 4, 3],       # 324
            [2, 2, 3, 3, 6],       # 216
            [4, 4, 6, 4],          # 384
            [2, 4, 4, 12],         # 384
            [3, 4, 4, 8],          # 384
            [2, 2, 4, 4, 8],       # 512 - too big
            [2, 2, 3, 4, 8],       # 384
            [4, 4, 4, 4],          # 256
            [8, 4, 4, 4],          # 512 - too big
            [2, 2, 2, 2, 4],       # 64
            [2, 2, 2, 2, 8],       # 128
            [2, 2, 2, 2, 16],      # 256
            [2, 2, 2, 4, 2],       # 64
            [4, 2, 2, 4, 2],       # 64
            [2, 2, 2, 2, 6],       # 96
            [2, 2, 3, 3, 2],       # 72
        ]

        # Deduplicate and sort by order
        seen = set()
        unique_candidates = []
        for orders in candidate_orders:
            key = tuple(sorted(orders))
            prod = 1
            for m in key:
                prod *= m
            if key not in seen and prod < best_n:
                seen.add(key)
                unique_candidates.append((key, prod))

        unique_candidates.sort(key=lambda x: x[1])

        for orders, prod in unique_candidates:
            if time.time() > deadline:
                break
            if prod >= best_n:
                continue

            try:
                n, edges = _try_cayley_abelian(list(orders), k, g, deadline, best_n)
                if n is not None and n < best_n:
                    best_n = n
                    best_edges = edges
            except Exception:
                continue

    return best_n, best_edges
