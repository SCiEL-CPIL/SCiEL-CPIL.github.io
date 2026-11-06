"""Clique partitioning via exact maximum-clique extraction on the sparse complement graph.

Structural approach:
  - G is dense (p=0.6 / 0.55). Its complement H = missing-edge graph is SPARSE.
  - A clique in G  <=>  an independent set in H.
  - Partition G's edges into cliques <=> cover G's edges with independent sets of H (size>=2).
  - Each step: find the LARGEST clique in the current "remaining G" (i.e. largest
    independent set in the current "remaining H") using branch-and-bound, remove its
    edges, repeat. Taking the maximum clique each step minimizes the number of cliques.
"""
import time
from random import Random


def _random_graph(n, p, seed):
    """Generate random graph using the same RNG as the verifier (bitmask adjacency of G)."""
    rng = Random(seed)
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj


def _complement(adj, n):
    """Return bitmask adjacency of the complement graph H (missing edges of G)."""
    h = [0] * n
    for u in range(n):
        full = 0
        for v in range(n):
            if u != v:
                full |= (1 << v)
        h[u] = full & ~adj[u]
    return h


def _max_independent_set(h, n, deadline):
    """
    Branch-and-bound maximum independent set in graph h (bitmask adjacency, size n).
    Returns a list of vertices forming a maximum (or near-maximum) independent set.
    Uses a greedy coloring upper bound (Tomita style) adapted for independent sets:
    a vertex's 'color' bound = 1 + bound of its non-neighbors... we instead use the
    standard clique B&B on the complement of h (which is G restricted), but here we
    directly do independent-set B&B with a degree-based bound.
    """
    best = []
    best_size = 0

    # Precompute: for speed, we work on the subgraph induced by `cand` (bitset).
    # Bound: number of vertices in cand is an upper bound; we refine with a greedy
    # independent-set size as a *lower* bound and use a simple upper bound = popcount.
    # For a stronger bound we use: a vertex can be in the set only with its non-neighbors.
    # We use a coloring of cand's complement (i.e., clique bound on cand's complement)
    # which equals the independent-set bound on cand. Simpler: use greedy coloring on
    # the graph `cand` itself to bound the *clique* of cand's complement = independent set of cand.

    def color_bound(cand):
        """Upper bound on max independent set size in subgraph induced by cand.
        We greedily color the COMPLEMENT of cand's induced subgraph; the number of
        colors is an upper bound on the clique of the complement = independent set of cand.
        Equivalently, greedily color cand's induced subgraph: each color class is an
        independent set in cand's complement, so a clique in cand's complement (i.e. an
        independent set in cand) uses at most (number of color classes) ... 
        Standard: max clique in graph X <= number of colors in a proper coloring of X.
        Here we want max independent set in cand = max clique in complement(cand).
        So color complement(cand) and use #colors as bound.
        """
        # Build list of vertices in cand
        verts = []
        t = cand
        while t:
            low = t & (-t)
            verts.append(low.bit_length() - 1)
            t ^= low
        # Color the complement of cand: edge in complement if NOT adjacent in h.
        # Greedy coloring: assign smallest color such that no already-colored vertex
        # with same color is a NON-neighbor in h (i.e., they are adjacent in complement).
        color_of = {}
        next_color = 0
        for v in verts:
            used = set()
            for w in verts:
                if w == v:
                    continue
                c = color_of.get(w)
                if c is None:
                    continue
                # v and w are adjacent in complement(cand) iff NOT adjacent in h
                if not (h[v] >> w) & 1:
                    used.add(c)
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            if c + 1 > next_color:
                next_color = c + 1
        return next_color

    def expand(cand, current):
        nonlocal best, best_size
        if not cand:
            if len(current) > best_size:
                best_size = len(current)
                best = list(current)
            return
        # Bound
        ub = color_bound(cand)
        if len(current) + ub <= best_size:
            return
        if time.time() > deadline:
            return
        # Pick a vertex: choose one with fewest neighbors in cand (to prune)
        # Actually for independent set we want to branch on a vertex; include or exclude.
        # Choose vertex v in cand. Branch: include v (then cand &= ~neighbors_h[v]) 
        # or exclude v (cand &= ~(1<<v)).
        # To get good bounds, pick v with high degree in cand (few non-neighbors) so
        # including it prunes a lot.
        t = cand
        best_v = -1
        best_deg = -1
        while t:
            low = t & (-t)
            v = low.bit_length() - 1
            deg = (h[v] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            t ^= low
        v = best_v
        # Include v
        current.append(v)
        expand(cand & ~(h[v] | (1 << v)), current)
        current.pop()
        # Exclude v
        expand(cand & ~(1 << v), current)

    all_verts = (1 << n) - 1
    # Remove isolated-in-G? No. Start with all vertices.
    # But we only care about independent sets of size >=2 that correspond to cliques in G
    # with all edges still present. The caller ensures cand is restricted appropriately.
    expand(all_verts, [])
    return best


def _solve_instance(adj, n, deadline):
    """Partition G's edges into cliques. Returns list of cliques (vertex lists)."""
    h = _complement(adj, n)
    # remaining edges of G as bitmasks
    rem = [adj[i] for i in range(n)]
    remaining_edges = sum(rem[i].bit_count() for i in range(n)) // 2
    cliques = []

    if remaining_edges == 0:
        return cliques

    while remaining_edges > 0:
        if time.time() > deadline:
            break

        # Build the "remaining complement" h_rem: an edge (u,v) is present in h_rem
        # iff it is a missing edge in G AND ... actually independent set in h_rem must
        # be a clique in the REMAINING G. The remaining G has edge (u,v) iff rem has it.
        # So a set S is a clique in remaining G iff for all u,v in S, rem[u] has bit v.
        # Equivalently S is an independent set in the graph h_rem where h_rem has edge
        # (u,v) iff NOT (rem[u] has bit v) [and u!=v]. That is, h_rem = complement of
        # remaining G.
        # We can compute h_rem on the fly: h_rem[u] = (full) & ~rem[u].
        full = 0
        for v in range(n):
            full |= (1 << v)

        # Find max independent set in h_rem restricted to vertices that have >=1 remaining edge.
        active = 0
        for i in range(n):
            if rem[i]:
                active |= (1 << i)

        # We need a B&B that uses h_rem (complement of rem) instead of h.
        # Reuse _max_independent_set by building h_rem adjacency.
        h_rem = [0] * n
        for u in range(n):
            h_rem[u] = (full & ~rem[u]) & ~(1 << u) & active
        # Also mask out inactive vertices
        for u in range(n):
            h_rem[u] &= active

        # Run B&B on h_rem over the `active` vertex set.
        # _max_independent_set uses global h; instead we call a local version.
        clique = _max_independent_set_on(h_rem, n, active, deadline)
        if not clique or len(clique) < 2:
            # Fallback: take a single edge
            # find any remaining edge
            found = False
            for i in range(n):
                t = rem[i]
                while t:
                    low = t & (-t)
                    j = low.bit_length() - 1
                    if j > i:  # ensure i<j to avoid dup
                        cliques.append([i, j])
                        rem[i] &= ~(1 << j)
                        rem[j] &= ~(1 << i)
                        remaining_edges -= 1
                        found = True
                        break
                    t ^= low
                if found:
                    break
            if not found:
                # shouldn't happen
                break
            continue

        # Remove all edges of the clique from rem
        m = len(clique)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = clique[i], clique[j]
                rem[a] &= ~(1 << b)
                rem[b] &= ~(1 << a)
        cliques.append(clique)
        remaining_edges -= m * (m - 1) // 2

    # Handle leftover edges (deadline hit)
    if remaining_edges > 0:
        for i in range(n):
            if time.time() > deadline:
                break
            t = rem[i]
            while t:
                if time.time() > deadline:
                    break
                low = t & (-t)
                j = low.bit_length() - 1
                if j > i:
                    cliques.append([i, j])
                    rem[i] &= ~(1 << j)
                    rem[j] &= ~(1 << i)
                t ^= low

    return cliques


def _max_independent_set_on(h, n, active, deadline):
    """Branch-and-bound max independent set in graph h restricted to `active` vertices."""
    best = []
    best_size = 0

    def color_bound(cand):
        verts = []
        t = cand
        while t:
            low = t & (-t)
            verts.append(low.bit_length() - 1)
            t ^= low
        color_of = {}
        next_color = 0
        for v in verts:
            used = set()
            for w in verts:
                if w == v:
                    continue
                c = color_of.get(w)
                if c is None:
                    continue
                if not ((h[v] >> w) & 1):
                    used.add(c)
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            if c + 1 > next_color:
                next_color = c + 1
        return next_color

    def expand(cand, current):
        nonlocal best, best_size
        if not cand:
            if len(current) > best_size:
                best_size = len(current)
                best = list(current)
            return
        ub = color_bound(cand)
        if len(current) + ub <= best_size:
            return
        if time.time() > deadline:
            return
        # pick vertex with max degree in cand
        t = cand
        best_v = -1
        best_deg = -1
        while t:
            low = t & (-t)
            v = low.bit_length() - 1
            deg = (h[v] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            t ^= low
        v = best_v
        current.append(v)
        expand(cand & ~(h[v] | (1 << v)), current)
        current.pop()
        expand(cand & ~(1 << v), current)

    expand(active, [])
    return best


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            results.append([])
            continue
        adj = _random_graph(n, p, seed)
        cliques = _solve_instance(adj, n, deadline)
        results.append(cliques)
    return results
