"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

sys.setrecursionlimit(1000000)

_DEADLINE = time.time() + 130.0  # safety margin before the 150s kill


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    for k in list(adj.keys()):
        adj[k] = list(adj[k])
    return adj


def _greedy(adj, edges, rng, time_budget):
    """Repeatedly extract the longest path from a random low-degree start."""
    remaining = set(edges)
    paths = []
    while remaining:
        if time.time() > time_budget:
            return None
        # candidate start vertices: endpoints of remaining edges
        cand = set()
        for (u, v) in remaining:
            cand.add(u)
            cand.add(v)
        # prefer low degree (fewer remaining incident edges)
        def deg(x):
            return sum(1 for (a, b) in remaining if a == x or b == x)
        # sample a few low-degree candidates
        cand_list = list(cand)
        cand_list.sort(key=deg)
        k = min(len(cand_list), 4)
        starts = cand_list[:k]
        rng.shuffle(starts)

        best_path = None
        best_len = -1
        for s in starts:
            # DFS longest path from s over remaining edges
            stack = [(s, [s], [s])]
            found = None
            found_len = 0
            while stack:
                node, path, used_edges = stack.pop()
                # try to extend
                nxts = [nb for nb in adj[node]
                        if (min(node, nb), max(node, nb)) in remaining and nb not in path]
                if not nxts:
                    if len(path) > found_len:
                        found = path
                        found_len = len(path)
                else:
                    # order neighbors by degree (prefer low-degree to keep options open)
                    nxts.sort(key=lambda x: deg(x))
                    for nb in nxts:
                        ne = (min(node, nb), max(node, nb))
                        stack.append((nb, path + [nb], used_edges + [ne]))
            if found is not None and len(found) > best_len:
                best_len = len(found)
                best_path = found
        if best_path is None or best_len < 2:
            # fallback: single edge
            e = next(iter(remaining))
            best_path = list(e)
        # remove edges of this path
        for i in range(len(best_path) - 1):
            a, b = best_path[i], best_path[i + 1]
            remaining.discard((min(a, b), max(a, b)))
        paths.append(best_path)
    return paths


def _try_decompose(edges, adj, rng, time_budget):
    paths = _greedy(adj, edges, rng, time_budget)
    if paths is None:
        return None
    # validate
    used = set()
    for p in paths:
        if len(p) < 1:
            return None
        seen = set(p)
        if len(seen) != len(p):
            return None  # not simple
        for i in range(len(p) - 1):
            a, b = p[i], p[i + 1]
            e = (min(a, b), max(a, b))
            if e not in edges:
                return None
            if e in used:
                return None
            used.add(e)
    if used != set(edges):
        return None
    return paths


def construct():
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    instances = [_instance(*s) for s in specs]
    adjs = [_build_adj(e) for e in instances]

    results = [None] * 3
    # per-instance time budget
    per_budget = 40.0
    total_budget = time.time() + per_budget * 3

    for idx in range(3):
        edges = instances[idx]
        adj = adjs[idx]
        best = None
        best_count = len(edges)  # trivial bound
        t_start = time.time()
        limit = min(t_start + per_budget, total_budget - 5.0)
        # restart with different seeds
        attempt = 0
        while time.time() < limit:
            rng = random.Random(1000 + attempt)
            paths = _try_decompose(edges, adj, rng, limit)
            if paths is not None:
                cnt = len(paths)
                if cnt < best_count:
                    best_count = cnt
                    best = paths
            attempt += 1
        if best is None:
            # fallback: every edge as its own path
            best = [list(e) for e in edges]
        results[idx] = best

    return results
# EVOLVE-BLOCK-END
