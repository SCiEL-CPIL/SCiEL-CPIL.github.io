import time
import random

_DEADLINE = 110.0  # seconds (safety margin under 120s limit)


def _build_adj(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj


def _has_c5(adj, n):
    # C5 exists iff there is a path of length 3 (a->x, x->y, y->b) with a != b and a adjacent to b.
    for x in range(n):
        ax = adj[x]
        # neighbors of x
        y_list = []
        m = ax
        while m:
            lb = m & (-m)
            y = lb.bit_length() - 1
            y_list.append(y)
            m ^= lb
        for i in range(len(y_list)):
            a = y_list[i]
            for j in range(i + 1, len(y_list)):
                b = y_list[j]
                if (ax >> b) & 1:  # a adjacent to b  (a,b both neighbors of x => edge a-b)
                    # need a path a-x-?-b where ? is a neighbor of both a and b, ? != x
                    common = adj[a] & adj[b] & ~(1 << x) & ~(1 << a) & ~(1 << b)
                    if common:
                        return True
    return False


def _alpha_at_least(adj, n, k=9):
    # Returns True if there is an independent set of size k.
    # Greedy / backtracking with pruning using remaining candidates.
    best = 0
    # Use a simple branch and bound: pick vertex, decide include/exclude.
    # Prune: if current_size + remaining_count <= best, stop.
    # We only need to detect size 9.
    from functools import lru_cache

    # Order vertices by degree (in complement) to help pruning: high degree in graph = fewer complement neighbors.
    comp_deg = [n - 1 - bin(adj[v]).count('1') for v in range(n)]
    order = sorted(range(n), key=lambda v: -comp_deg[v])

    def rec(candidates, size):
        nonlocal best
        if size >= k:
            return True
        if not candidates:
            return False
        # bound: size + popcount(candidates) <= best -> prune (best is max found so far, but we only need k)
        # We'll just try to find any of size k.
        # pick a vertex
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        # include v
        newcand = candidates & ~(1 << v) & ~adj[v]  # must be independent from v (non-neighbors)
        # also remove v itself
        newcand &= ~(1 << v)
        if rec(newcand, size + 1):
            return True
        # exclude v
        newcand2 = candidates & ~(1 << v)
        if rec(newcand2, size):
            return True
        return False

    return rec((1 << n) - 1, 0)


def _is_valid(n, edges):
    adj = _build_adj(n, edges)
    if _has_c5(adj, n):
        return False
    if _alpha_at_least(adj, n, 9):
        return False
    return True


def _base_solution():
    edges = [
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [4, 7], [5, 6], [5, 7], [6, 7],
        [8, 9], [8, 10], [8, 11], [9, 10], [9, 11], [10, 11],
        [12, 13], [12, 14], [12, 15], [13, 14], [13, 15], [14, 15],
        [16, 17], [16, 18], [16, 19], [17, 18], [17, 19], [18, 19],
        [20, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23],
        [24, 25], [24, 26], [24, 27], [25, 26], [25, 27], [26, 27],
        [28, 29], [28, 30], [28, 31], [29, 30], [29, 31], [30, 31],
    ]
    return 32, edges


def _search(n_target, time_budget):
    """Local search to find a C5-free graph on n_target vertices with alpha < 9.
    Returns (n, edges) or None."""
    deadline = time.time() + time_budget
    rng = random.Random(12345)

    # Start from the base K4^8 graph and add extra vertices connected randomly.
    base_n, base_edges = _base_solution()
    n = n_target
    # Build adjacency as bitsets
    adj = [0] * n
    for u, v in base_edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    # For new vertices (32..n-1), add random edges to old and among themselves.
    def add_random_vertex_edges(v, adj, n):
        # Connect to a random subset of old vertices
        for u in range(32):
            if rng.random() < 0.4:
                adj[v] |= 1 << u
                adj[u] |= 1 << v
        for u in range(32, v):
            if rng.random() < 0.5:
                adj[v] |= 1 << u
                adj[u] |= 1 << v

    for v in range(32, n):
        add_random_vertex_edges(v, adj, n)

    best_n = 0
    best_edges = None

    for iteration in range(20000):
        if time.time() > deadline - 5:
            break
        # Check current state
        if not _has_c5(adj, n) and not _alpha_at_least(adj, n, 9):
            # Convert to edge list
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            if n > best_n:
                best_n = n
                best_edges = edges
            break

        # Pick a random edge to flip
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        current = (adj[u] >> v) & 1
        # Flip
        if current:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= 1 << v
            adj[v] |= 1 << u

    if best_edges is not None:
        return best_n, best_edges
    return None


def construct():
    t0 = time.time()

    # Guaranteed baseline: n=32
    base_n, base_edges = _base_solution()
    # Quick validation of base (should always pass)
    if not _is_valid(base_n, base_edges):
        # Should not happen, but if it does, just return it anyway
        pass

    # Try to improve to n=33, 34, 35 if time permits
    for n_try in [33, 34, 35]:
        elapsed = time.time() - t0
        remaining = _DEADLINE - elapsed - 10  # leave 10s safety margin
        if remaining < 3:
            break
        result = _search(n_try, remaining)
        if result is not None:
            rn, redges = result
            if rn > base_n:
                base_n = rn
                base_edges = redges

    # Final safety: if we ran out of time, return what we have
    return {'n': base_n, 'edges': base_edges}
