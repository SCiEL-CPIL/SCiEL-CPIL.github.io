import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill


def _baseline_adj(n=16):
    """K_{8,8} plus one internal edge (0,1)."""
    adj = np.zeros((n, n), dtype=bool)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = adj[v, u] = True
    adj[0, 1] = adj[1, 0] = True
    return adj


def _score(adj):
    """Largest deg(u)+deg(v)+deg(w) over triangles. 0 if no triangle."""
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    best = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                common = adj[u] & adj[v]
                if not common.any():
                    continue
                ws = np.nonzero(common)[0]
                for w in ws:
                    s = deg[u] + deg[v] + deg[w]
                    if s > best:
                        best = s
    return int(best)


def _triangles(adj):
    n = adj.shape[0]
    tris = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                common = adj[u] & adj[v]
                for w in np.nonzero(common)[0]:
                    tris.append((u, v, w))
    return tris


def _edge_count(adj):
    return int(adj.sum() // 2)


def construct():
    n = 16
    target = 65

    best_adj = _baseline_adj(n)
    best_score = _score(best_adj)
    rng = random.Random(12345)

    # Local search: add an edge, then remove a different edge to keep count.
    # Try to reduce the maximum triangle degree-sum.
    # Precompute all non-edges and edges for candidate generation.

    def neighbors_of(a, b):
        return np.nonzero(adj_ref[a] & adj_ref[b])[0]

    # We'll do hill-climbing with occasional restarts if stuck.
    iters = 0
    max_iters = 200000
    while time.time() < _DEADLINE and iters < max_iters:
        iters += 1
        # Build candidate edge sets (add e1, remove e2, e1 != e2, both valid)
        # To be fast, pick a random non-edge to add and a random edge to remove.
        # But smarter: focus on edges that participate in high-sum triangles.

        # Get current triangles and their sums
        deg = best_adj.sum(axis=1)
        tris = _triangles(best_adj)
        if not tris:
            # No triangles; any swap keeps it triangle-free? Not necessarily.
            # Just try a random swap.
            pass

        # Identify "bad" triangles (high sum)
        # We'll try to break the worst triangle by removing one of its edges.
        if tris:
            # Sort by sum desc
            tris_sorted = sorted(tris, key=lambda t: deg[t[0]] + deg[t[1]] + deg[t[2]], reverse=True)
            # Try removing an edge from the worst triangle(s)
            candidates = []
            seen = set()
            for (u, v, w) in tris_sorted[:30]:
                for (a, b) in [(u, v), (v, w), (u, w)]:
                    if (a, b) in seen:
                        continue
                    seen.add((a, b))
                    if best_adj[a, b]:
                        candidates.append((a, b))

        if not candidates:
            # Fallback: random edge
            edges_list = np.argwhere(best_adj)
            edges_list = [(int(r[0]), int(r[1])) for r in edges_list if r[0] < r[1]]
            candidates = edges_list

        # Pick a remove candidate
        a, b = candidates[rng.randrange(len(candidates))]

        # Pick an add candidate: a non-edge, not equal to (a,b)
        # Prefer non-edges that don't create high-sum triangles, but keep it simple.
        # Generate non-edges
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n)
                     if not best_adj[i, j]]
        if not non_edges:
            break
        c, d = non_edges[rng.randrange(len(non_edges))]

        # Apply swap
        best_adj[a, b] = best_adj[b, a] = False
        best_adj[c, d] = best_adj[d, c] = True

        new_score = _score(best_adj)
        if new_score < best_score:
            best_score = new_score
        else:
            # Revert
            best_adj[c, d] = best_adj[d, c] = False
            best_adj[a, b] = best_adj[b, a] = True

        # Early exit if we reach the conjectured optimum
        if best_score <= 23:
            break

    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j]:
                edges.append([i, j])

    return {'n': n, 'edges': edges}
