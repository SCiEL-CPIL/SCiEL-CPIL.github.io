"""Different strategy: randomized local search with fast bitset validation."""
import time
import random
import sys

sys.setrecursionlimit(10000)

# Explicit deadline guard: evaluator kills at 150s wall-clock; keep safety margin.
_DEADLINE = 140.0
_start = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _start)

def _has_triangle(adj, n):
    """Fast triangle check using bitsets: for each edge (u,v), check adj[u]&adj[v] != 0."""
    for u in range(n):
        du = adj[u]
        # Iterate over neighbors of u
        v = u + 1
        while v < n:
            # find next set bit >= v
            if du >> v & 1:
                if du & adj[v]:
                    return True
            v += 1
    return False

def _max_independent_set(adj, n):
    """Exact maximum independent set via branch and bound (Bron-Kerbosch style on complement)."""
    # We want largest set S with no edges among S in G, i.e., clique in complement.
    # Build complement adjacency (non-edges of G, excluding self).
    comp = [0] * n
    for i in range(n):
        # all bits except i and adj[i]
        full = (1 << n) - 1
        comp[i] = full & ~(adj[i] | (1 << i))

    best = 0

    def expand(R_size, P):
        nonlocal best
        if P == 0:
            if R_size > best:
                best = R_size
            return
        # Bound: R_size + popcount(P) <= best -> prune
        if R_size + bin(P).count('0') - bin(P).count('1') <= best:
            # popcount of P
            pc = P.bit_count()
            if R_size + pc <= best:
                return
        # Simple greedy bound: use a vertex ordering
        # Choose vertex with max degree in P for branching (or min for speed)
        # Pick a vertex v in P
        # Use a heuristic: pick v that maximizes |P & comp[v]|
        Pbits = P
        # To keep it simple and fast, pick lowest set bit
        v = (Pbits & -Pbits).bit_length() - 1
        # Branch: include v
        P2 = P & comp[v]
        expand(R_size + 1, P2)
        # Branch: exclude v
        P3 = P & ~(1 << v)
        expand(R_size, P3)

    expand(0, (1 << n) - 1)
    return best

def _validate(adj, n):
    """Check triangle-free and alpha < 10."""
    if _has_triangle(adj, n):
        return False
    alpha = _max_independent_set(adj, n)
    return alpha < 10

def _random_triangle_free(n):
    """Generate a random triangle-free graph via random ordering and only adding edges that don't create triangles."""
    adj = [0] * n
    order = list(range(n))
    random.shuffle(order)
    for i in range(n):
        u = order[i]
        for j in range(i + 1, n):
            v = order[j]
            if random.random() < 0.35:
                # Check if adding (u,v) creates a triangle
                # A triangle would need a common neighbor w of u and v already in graph
                if adj[u] & adj[v] == 0:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
    return adj

def _local_search(adj, n, max_iters=20000):
    """Local search: try to reduce alpha by adding edges, while maintaining triangle-free.
    Strategy: find a large independent set, then try to add edges within it that don't create triangles."""
    alpha = _max_independent_set(adj, n)
    if alpha < 10:
        return adj, True

    iters = 0
    while iters < max_iters and _time_left() > 5:
        iters += 1
        # Find an independent set of size alpha (or near)
        # We'll do a greedy independent set
        indep = []
        remaining = (1 << n) - 1
        while remaining:
            # Pick vertex with min degree in remaining (greedy)
            v = (remaining & -remaining).bit_length() - 1
            indep.append(v)
            remaining &= ~(1 << v)
            remaining &= ~adj[v]

        if len(indep) < 10:
            # Try to add edges within indep to break it
            for i in range(len(indep)):
                for j in range(i + 1, len(indep)):
                    u, v = indep[i], indep[j]
                    if adj[u] & (1 << v) == 0:  # not already edge
                        if adj[u] & adj[v] == 0:  # no common neighbor
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u
                            # Check if this reduces alpha
                            new_alpha = _max_independent_set(adj, n)
                            if new_alpha < alpha:
                                alpha = new_alpha
                                if alpha < 10:
                                    return adj, True
                            else:
                                # Revert
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
        else:
            # Try random edge addition within the independent set
            if len(indep) >= 2:
                i, j = random.sample(range(len(indep)), 2)
                u, v = indep[i], indep[j]
                if adj[u] & (1 << v) == 0:
                    if adj[u] & adj[v] == 0:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        new_alpha = _max_independent_set(adj, n)
                        if new_alpha < alpha:
                            alpha = new_alpha
                            if alpha < 10:
                                return adj, True
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
        if alpha < 10:
            return adj, True

    return adj, _validate(adj, n)

def construct():
    global _start
    _start = time.time()
    
    # Start with Andrasfai And(9) as guaranteed baseline
    n_base = 26
    adj_base = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if (j - i) % n_base % 3 == 1:
                adj_base[i] |= 1 << j
                adj_base[j] |= 1 << i
    best_adj = adj_base[:]
    best_n = n_base

    # Try to find larger valid graphs via random search + local search
    for n in [28, 30, 32, 34, 36, 38, 39]:
        if _time_left() < 10:
            break
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and _time_left() > 5:
            attempts += 1
            adj = _random_triangle_free(n)
            adj, valid = _local_search(adj, n, max_iters=10000)
            if valid:
                best_adj = adj
                best_n = n
                break  # found valid for this n, try larger
            if _time_left() < 10:
                break

    # Convert best_adj to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
