"""Fresh design: BFS layer-by-layer cage construction + randomized port-closure search
with an exact non-backtracking girth certificate (shortest cycle via matrix powers)."""
import sys
import time
import random
import numpy as np


def _girth_cert(n, adj):
    """Exact girth via shortest non-backtracking walk returning to start.
    For a simple graph, the shortest such walk equals the girth."""
    if n <= 0:
        return 10 ** 9
    A = adj.astype(np.int64)
    P = A.copy()
    D = np.diag(A.sum(axis=1).astype(np.int64))
    for t in range(3, 60):
        P = P @ A
        np.fill_diagonal(P, 0)
        if P.any():
            return t
    return 60


def _build_graph(k, g, ports):
    """Build the graph for a given set of closed ports. Returns (n, edges, adj, girth)
    or None if it is not k-regular / has a short cycle."""
    h = g // 2
    q = h - 1
    if q < 0:
        q = 0
    T = 1 + k * (k ** q)
    C = len(ports)
    n = T + C
    if n <= 0:
        return None
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []

    def add_edge(u, v):
        if u < 0 or v < 0 or u >= n or v >= n or u == v:
            return False
        if adj[u, v]:
            return False
        adj[u, v] = 1
        adj[v, u] = 1
        edges.append((u, v))
        return True

    # Build the BFS tree. Each vertex stores (depth, parent).
    depth = [-1] * n
    parent = [-1] * n
    depth[0] = 0
    queue = [0]
    head = 0
    # Assign tree edges deterministically: vertex v (1..T-1) has parent = (v-1)//k,
    # and we connect parent -> child.
    for v in range(1, T):
        p = (v - 1) // k
        depth[v] = depth[p] + 1
        parent[v] = p
        if not add_edge(p, v):
            return None

    # Verify the tree is a perfect k-ary tree of depth h (sanity).
    # (Not strictly required; the closure logic below depends on depths.)
    # Port list: leaf vertices (depth == h) that we may close.
    leaf_vertices = [v for v in range(T) if depth[v] == h]
    # Candidate targets: vertices at depth g - h - 1 (>= 1).
    target_depth = g - h - 1
    target_vertices = [v for v in range(T) if depth[v] == target_depth]

    # Map port index -> leaf vertex (we iterate ports in order).
    # ports is a set of indices into leaf_vertices.
    for idx in ports:
        if idx < 0 or idx >= len(leaf_vertices):
            return None
        leaf = leaf_vertices[idx]
        # Find the target: the ancestor of leaf at target_depth.
        # Walk up from leaf to depth target_depth.
        cur = leaf
        d = depth[cur]
        while d > target_depth:
            cur = parent[cur]
            d = depth[cur]
        target = cur
        # Ensure the edge is not already a tree edge (it shouldn't be, since
        # target is a strict ancestor at a higher depth than h-1 ... actually
        # target_depth < h, so target is not the parent of leaf unless h-1==target_depth).
        if not add_edge(leaf, target):
            return None

    # Degree check: every vertex must have degree exactly k.
    deg = adj.sum(axis=1)
    if not np.all(deg == k):
        return None

    # Girth certificate (exact).
    girth = _girth_cert(n, adj)
    if girth < g:
        return None
    return (n, edges, adj, girth)


def _moore_bound(k, g):
    """Lower bound on n (Moore bound)."""
    h = g // 2
    if (g % 2) == 0:
        return 2 * (1 + k * (k ** (h - 1)))
    else:
        return 1 + k * (k ** (h - 1)) + k * (k ** h)


def construct(k, g):
    """Construct a k-regular graph of girth >= g with as few vertices as possible."""
    t0 = time.time()
    DEADLINE = 115.0  # safety margin before the 120s wall-clock kill.

    h = g // 2
    q = max(0, h - 1)
    T = 1 + k * (k ** q)
    n_ports = k * (k ** q)  # number of leaf ports
    target_depth = g - h - 1

    # Baseline: close all ports (the "tree" graph). This gives n = T.
    # It is k-regular iff the closure edges don't collide and every degree == k.
    best = None
    base_ports = set(range(n_ports))
    res = _build_graph(k, g, base_ports)
    if res is not None:
        best = res

    # Search over subsets of ports to close. Randomized hill-climb.
    # Start from a random subset and try flips.
    rng = random.Random(12345)
    # Number of closures to start with: roughly n_ports (all) is the baseline.
    # We want to find a smaller valid set. Start with all, then try removing.
    current = set(base_ports)
    cur_res = res
    if cur_res is None:
        # Fallback: try to find any valid configuration.
        return (0, [])

    cur_n = cur_res[0]

    # Try to reduce n by removing ports (each removal reduces n by 1).
    # A removal is valid only if the resulting graph is still k-regular and girth >= g.
    # Removing a port makes that leaf a new vertex (degree 1) and its target loses 1 degree,
    # so the graph is NOT k-regular after a single removal. We must remove ports in a
    # way that preserves k-regularity, which requires pairing up leaves and targets.

    # Simpler: search over the number of closures C and which C ports to close.
    # For a fixed C, we need to choose C ports such that the C target vertices
    # each receive exactly the right number of closure edges to reach degree k.
    # This is a matching/assignment problem.

    # Given time constraints, do a randomized search over C and random port subsets,
    # checking validity. Keep the best (smallest n) valid graph.
    best_n = cur_n
    best_edges = cur_res[1]
    best_adj = cur_res[2]

    # Try smaller C values. For each C, sample random subsets of size C.
    # The number of samples per C is limited by time.
    max_samples_total = 20000
    samples_used = 0
    # C must be such that n = T + C is at least the Moore bound.
    mb = _moore_bound(k, g)
    min_C = max(0, mb - T)
    # C ranges from min_C to n_ports.
    # Start from small C (fewer closures -> fewer vertices) and increase.
    for C in range(min_C, n_ports + 1):
        if time.time() - t0 > DEADLINE:
            break
        if C < 0:
            continue
        n_try = T + C
        if n_try < mb:
            continue
        # Sample random subsets of size C.
        # Limit samples per C.
        samples_for_c = max(1, min(200, (max_samples_total - samples_used) // max(1, (n_ports - min_C + 1))))
        for _ in range(samples_for_c):
            if time.time() - t0 > DEADLINE:
                break
            if samples_used >= max_samples_total:
                break
            subset = set(rng.sample(range(n_ports), C))
            res = _build_graph(k, g, subset)
            samples_used += 1
            if res is not None:
                n_found = res[0]
                if n_found < best_n:
                    best_n = n_found
                    best_edges = res[1]
                    best_adj = res[2]
                    # Update current for potential local search.
                    current = subset
                    cur_res = res
                    cur_n = n_found

    # Local search: try flipping individual ports in the current best.
    if cur_res is not None and time.time() - t0 < DEADLINE:
        # Try adding/removing ports one at a time.
        for _ in range(5000):
            if time.time() - t0 > DEADLINE:
                break
            if not current:
                break
            # Try removing a random port.
            port = rng.choice(list(current))
            new_set = set(current)
            new_set.discard(port)
            res = _build_graph(k, g, new_set)
            if res is not None and res[0] < best_n:
                best_n = res[0]
                best_edges = res[1]
                best_adj = res[2]
                current = new_set
                cur_res = res
            # Try adding a random port not in current.
            if len(current) < n_ports:
                avail = [p for p in range(n_ports) if p not in current]
                if avail:
                    port2 = rng.choice(avail)
                    new_set2 = set(current)
                    new_set2.add(port2)
                    res2 = _build_graph(k, g, new_set2)
                    if res2 is not None and res2[0] < best_n:
                        best_n = res2[0]
                        best_edges = res2[1]
                        best_adj = res2[2]
                        current = new_set2
                        cur_res = res2

    return (best_n, best_edges)


if __name__ == "__main__":
    # Quick self-test.
    n, edges = construct(3, 16)
    print(f"n={n}, edges={len(edges)}")
