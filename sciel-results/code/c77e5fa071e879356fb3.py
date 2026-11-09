import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 135.0  # safety margin vs 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj

    def try_path(adj, start, visited_edges, visited_vertices, deadline):
        """Greedy: extend from start to a maximal simple path (no vertex reuse)."""
        path = [start]
        used_v = {start}
        cur = start
        while time.time() < deadline:
            nbrs = adj.get(cur, ())
            best = None
            # prefer unvisited neighbours, then fewest remaining options (dead-end first)
            cands = []
            for nb in nbrs:
                if (min(cur, nb), max(cur, nb)) in visited_edges:
                    continue
                if nb in used_v:
                    continue
                cands.append(nb)
            if not cands:
                break
            # deterministic-ish greedy: pick the neighbour that leaves the least
            # flexibility (minimum number of still-usable neighbours)
            def flex(nb):
                c = 0
                for m in adj.get(nb, ()):
                    if (min(nb, m), max(nb, m)) not in visited_edges and m not in used_v:
                        c += 1
                return c
            cands.sort(key=lambda x: (flex(x), x))
            cur = cands[0]
            path.append(cur)
            used_v.add(cur)
            visited_edges.add((min(path[-2], cur), max(path[-2], cur)))
        return path

    def decompose(edges, deadline):
        adj = build_adj(edges)
        remaining = set(edges)
        deg = {v: 0 for v in adj}
        for u, v in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1

        paths = []
        # process vertices by descending degree for better packing
        order = sorted(adj.keys(), key=lambda v: (-deg.get(v, 0), v))
        while remaining and time.time() < deadline:
            # pick the remaining edge whose endpoints have highest degree
            best_edge = None
            best_key = None
            for e in remaining:
                u, v = e
                key = (deg.get(u, 0) + deg.get(v, 0), u, v)
                if best_key is None or key > best_key:
                    best_key = key
                    best_edge = e
            if best_edge is None:
                break
            u, v = best_edge
            # try both orientations
            for start in (u, v):
                visited_edges = set()
                path = try_path(adj, start, visited_edges, set(), deadline)
                # check the path is a valid simple path using only edges
                if len(path) >= 2 and all(
                    (min(a, b), max(a, b)) in remaining for a, b in zip(path, path[1:])
                ) and len(set(path)) == len(path):
                    for a, b in zip(path, path[1:]):
                        remaining.discard((min(a, b), max(a, b)))
                    paths.append(path)
                    break
            else:
                # fallback: single edge
                remaining.discard((u, v))
                paths.append([u, v])
        return paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        if time.time() > DEADLINE - 5:
            # not enough time: return trivial per-edge paths for this and rest
            result.append([list(e) for e in edges])
            continue
        paths = decompose(edges, DEADLINE - 5)
        # safety: ensure we covered all edges; if not, add missing as singletons
        used = set()
        for p in paths:
            for a, b in zip(p, p[1:]):
                used.add((min(a, b), max(a, b)))
        for e in edges:
            if e not in used:
                paths.append(list(e))
        result.append(paths)
    return result
# EVOLVE-BLOCK-END
