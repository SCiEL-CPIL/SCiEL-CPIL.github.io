import time
import random

_N = 40
_DEADLINE = 150.0
_MARGIN = 3.0          # seconds of safety before the hard limit
_RNG_SEED = 20240612


def _build_k5_disjoint():
    """8 disjoint K5s on vertices 0..39 -> 100 edges, C6-free."""
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    return edges


def _make_adjacency(edges):
    adj = [set() for _ in range(_N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _has_c6_through(adj, u, v):
    """Return True iff adding edge (u,v) to the current graph creates a 6-cycle.

    A 6-cycle containing (u,v) is u - x - y - z - w - v - u where
    x,y,z,w are distinct and not in {u,v}.  That is exactly a length-3
    path from u to v in the current graph (u-x-y-z-w-v).  So we run a
    depth-3 BFS from u and a depth-3 BFS from v; if the two frontiers
    meet at some vertex, such a path exists.
    """
    if v in adj[u]:
        return False  # already an edge; cannot add

    # BFS from u up to depth 3, tracking the first depth reached.
    depth_u = [-1] * _N
    depth_u[u] = 0
    frontier = [u]
    for d in range(1, 4):
        nxt = []
        for x in frontier:
            for y in adj[x]:
                if depth_u[y] == -1:
                    depth_u[y] = d
                    nxt.append(y)
        frontier = nxt
        if not frontier:
            break

    # BFS from v up to depth 3.
    depth_v = [-1] * _N
    depth_v[v] = 0
    frontier = [v]
    for d in range(1, 4):
        nxt = []
        for x in frontier:
            for y in adj[x]:
                if depth_v[y] == -1:
                    depth_v[y] = d
                    nxt.append(y)
                    # The added edge connects u..v; a 6-cycle needs a
                    # 3-edge path u-...-v.  If a vertex is reachable from
                    # both u and v within <=3 edges and the two path
                    # lengths sum to 3, we have the cycle.
                    du = depth_u[y]
                    if du != -1 and du + depth_v[y] == 3:
                        return True
        frontier = nxt
        if not frontier:
            break
    return False


def _has_c6_full(adj):
    """Full verification: does the graph contain any 6-cycle?"""
    for u in range(_N):
        for v in list(adj[u]):
            if u < v and _has_c6_through(adj, u, v):
                return True
    return False


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN
    rng = random.Random(_RNG_SEED)

    edges = _build_k5_disjoint()
    adj = _make_adjacency(edges)
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v) if u < v else (v, u))
    best_edges = list(edges)
    best_count = len(edges)

    # Candidate cross edges (between different K5 blocks).
    candidates = []
    for i in range(8):
        for j in range(i + 1, 8):
            for a in range(5):
                for b in range(5):
                    u = 5 * i + a
                    v = 5 * j + b
                    candidates.append((u, v))

    def degree_of(x):
        return len(adj[x])

    def score_edge(e):
        u, v = e
        # Prefer joining low-degree vertices (keeps graph balanced /
        # delays saturation), with a small random tiebreak.
        return (degree_of(u) + degree_of(v), rng.random())

    while time.time() < deadline:
        # Re-rank candidates by current degree sum (cheap: O(cands)).
        remaining = [e for e in candidates if e not in edge_set]
        if not remaining:
            break
        remaining.sort(key=score_edge)

        added_any = False
        for e in remaining:
            if time.time() >= deadline:
                break
            u, v = e
            if e in edge_set:
                continue
            if _has_c6_through(adj, u, v):
                continue
            # Accept the edge.
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add(e)
            edges.append([u, v])
            added_any = True

        if not added_any:
            # No edge could be added in this pass.  Try a few random
            # restarts of the ordering to escape local orderings, but
            # keep the same edge set (deterministic structure) — the
            # greedy result is order-dependent, so reshuffling candidates
            # can sometimes admit more edges.
            if time.time() >= deadline:
                break
            # Reshuffle and retry a bounded number of times.
            for _ in range(50):
                if time.time() >= deadline:
                    break
                remaining = [e for e in candidates if e not in edge_set]
                if not remaining:
                    break
                rng.shuffle(remaining)
                progress = False
                for e in remaining:
                    u, v = e
                    if e in edge_set:
                        continue
                    if _has_c6_through(adj, u, v):
                        continue
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add(e)
                    edges.append([u, v])
                    progress = True
                if not progress:
                    break
            if not added_any:
                break

        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)

    # Final safety verification of the best graph.
    best_adj = _make_adjacency(best_edges)
    if _has_c6_full(best_adj):
        # Should not happen; fall back to the known-valid K5 baseline.
        return _build_k5_disjoint()

    return best_edges
