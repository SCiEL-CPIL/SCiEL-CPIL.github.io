# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

def _hilbert_index(x, y, bits=16):
    n = 1 << bits
    xs = (x * (n - 1)).astype(np.int64)
    ys = (y * (n - 1)).astype(np.int64)
    d = np.zeros(len(x), dtype=np.int64)
    s = 1 << (bits - 1)
    while s > 0:
        rx = (xs & s) > 0
        ry = (ys & s) > 0
        d += s * s * ((rx * 3) + ry)
        if ry == 0:
            if rx == 1:
                xs = n - 1 - xs
                ys = n - 1 - ys
            xs, ys = ys, xs
        s //= 2
    return d

def _mst_traversal_order(dm):
    """Get a traversal order from MST depth-first search."""
    n = dm.shape[0]
    # Build MST
    sparse_dm = csr_matrix(dm)
    mst = minimum_spanning_tree(sparse_dm).toarray()
    # Build adjacency
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0:
                adj[i].append(j)
    # DFS from node 0
    visited = [False] * n
    order = []
    stack = [0]
    visited[0] = True
    while stack:
        node = stack.pop()
        order.append(node)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
    return order

def _two_opt_limited(tour, dist, max_iters=100, deadline=None):
    n = len(tour)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if deadline is not None and time.time() > deadline:
            break
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                old = dist[a, b] + dist[c, d]
                new = dist[a, c] + dist[b, d]
                if new + 1e-12 < old:
                    tour = tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
                    improved = True
    return tour

def _greedy_tour(dist):
    n = dist.shape[0]
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = tour[-1]
        best_j = -1
        best_d = np.inf
        for j in range(n):
            if not visited[j] and dist[last, j] < best_d:
                best_d = dist[last, j]
                best_j = j
        tour.append(best_j)
        visited[best_j] = True
    return tour

def heuristics(distance_matrix):
    start_time = time.time()
    DEADLINE = 270.0

    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]

    eps = 1e-9
    base = 1.0 / (dm + eps)

    # MST-based traversal order for the spine
    try:
        spine_order = _mst_traversal_order(dm)
        spine_rank = np.empty(n, dtype=int)
        spine_rank[spine_order] = np.arange(n)
        spine_dist = np.abs(spine_rank[:, None] - spine_rank[None, :]).astype(float)
        max_spine = spine_dist.max() + 1e-12
        spine_norm = spine_dist / max_spine
        spine_mult = 1.0 / (1.0 + 4.0 * spine_norm)
    except Exception:
        spine_mult = np.ones((n, n))

    # Spine neighborhood boost: nearest 2 in spine order
    spine_boost = np.ones((n, n))
    try:
        for i in range(n):
            order_by_spine = np.argsort(spine_dist[i])
            order_by_spine = order_by_spine[order_by_spine != i]
            for j in order_by_spine[:2]:
                spine_boost[i, j] = 1.8
                spine_boost[j, i] = 1.8
    except Exception:
        pass

    heuristic = base * spine_mult * spine_boost

    # Reference tour: greedy + 2-Opt
    if time.time() < DEADLINE - 10.0:
        try:
            greedy = _greedy_tour(dm)
            refined = _two_opt_limited(greedy, dm, max_iters=50, deadline=DEADLINE - 10.0)
            tour_edge_mask = np.zeros((n, n))
            for k in range(n):
                a = refined[k]
                b = refined[(k + 1) % n]
                tour_edge_mask[a, b] = 1.0
                tour_edge_mask[b, a] = 1.0
            heuristic = heuristic * (1.0 + 0.4 * tour_edge_mask)
        except Exception:
            pass

    # Clean up
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    heuristic = (heuristic + heuristic.T) / 2.0
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
