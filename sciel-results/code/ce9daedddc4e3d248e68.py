"""Rewritten program: decompose each fixed instance graph into cycles + leftover edges."""
import time
import sys
from random import Random

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_longest_cycle(adj, max_depth):
    """Return a simple cycle (list of distinct vertices, len>=3) or None.
    Uses iterative-deepening DFS for longest simple path, then closes it."""
    n = len(adj)

    for depth in range(max_depth, 2, -1):
        # Try to find a simple path of exactly `depth` vertices whose
        # endpoints are adjacent (closing it into a cycle).
        best = None
        # Search from a few starting vertices to keep cost bounded.
        # Use a stack-based DFS for a path of length depth.
        # We look for path v0..v_{depth-1} with edge (v_{depth-1}, v0).
        # To make this tractable we do DFS building the path and at the
        # final depth check the closing edge.

        # We'll do a bounded DFS: explore paths up to `depth` vertices.
        # Track best (longest) path found that closes.
        found = None

        # Simple recursive DFS with pruning by remaining length.
        def dfs(start, path, depth_needed):
            # path is a list of vertices, last is current
            if depth_needed == 1:
                # Check closing edge: last vertex adjacent to start
                if start in adj[path[-1]]:
                    return path[:]
                return None
            cur = path[-1]
            # Order neighbors by degree (prefer low-degree to finish sooner)
            nbrs = sorted(adj[cur], key=lambda x: len(adj[x]))
            for nb in nbrs:
                if nb in path:
                    continue
                # Prune: even if we extend optimally we must reach depth_needed
                new_path = path + [nb]
                res = dfs(start, new_path, depth_needed - 1)
                if res is not None:
                    return res
            return None

        # Try a limited set of starting vertices (all, but cap work)
        # To bound runtime, try starts in degree order.
        starts = sorted(range(n), key=lambda x: len(adj[x]))
        for s in starts:
            if not adj[s]:
                continue
            res = dfs(s, [s], depth)
            if res is not None:
                found = res
                break
        if found is not None:
            return found
    return None


def _decompose(n, edges, deadline, time_budget):
    adj = _build_adjacency(n, edges)
    remaining = set(edges)
    pieces = []
    t0 = time.time()

    # Greedily extract long cycles.
    # Cap per-cycle search depth to keep runtime bounded.
    max_cycle_depth = min(n, 12)
    while remaining:
        if time.time() - t0 > time_budget:
            break
        # Try to find a long cycle in the current remaining graph.
        cycle = _find_longest_cycle(adj, max_cycle_depth)
        if cycle is None:
            break
        # Verify the cycle edges are all in remaining
        ok = True
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            e = (a, b) if a < b else (b, a)
            if e not in remaining:
                ok = False
                break
        if not ok:
            break
        # Commit cycle
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            e = (a, b) if a < b else (b, a)
            remaining.discard(e)
            adj[a].discard(b)
            adj[b].discard(a)
        pieces.append(cycle)

    # Leftover edges as single-edge pieces.
    for e in sorted(remaining):
        pieces.append(list(e))
    return pieces


def _validate(n, edges, pieces):
    seen = set()
    for piece in pieces:
        if len(piece) == 2:
            u, v = piece
            e = (u, v) if u < v else (v, u)
            if e in seen:
                return False
            seen.add(e)
        elif len(piece) >= 3:
            # cycle
            verts = piece
            if len(set(verts)) != len(verts):
                return False
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                e = (a, b) if a < b else (b, a)
                if e in seen:
                    return False
                seen.add(e)
        else:
            return False
    if seen != set(edges):
        return False
    return True


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        budget = max(1.0, min(40.0, deadline - time.time() - 5.0))
        try:
            pieces = _decompose(n, edges, deadline, budget)
            if not _validate(n, edges, pieces):
                pieces = [list(e) for e in edges]
        except Exception:
            pieces = [list(e) for e in edges]
        results.append(pieces)
    return results
