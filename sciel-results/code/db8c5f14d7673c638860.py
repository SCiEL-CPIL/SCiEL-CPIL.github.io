"""Decompose fixed instance graphs into few cycles + edges (Erdős-Gallai style)."""
import time
from collections import defaultdict, deque
from random import Random


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, n, time_limit=40.0):
    if not edges:
        return []

    adj = defaultdict(set)
    alive_edges = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        alive_edges.add((u, v))
        alive_edges.add((v, u))

    pieces = []
    start_time = time.time()
    deadline = start_time + time_limit

    def find_path_avoiding_edge(start, target, banned_edge, max_depth=10):
        """Find a path from start to target avoiding banned_edge, with length <= max_depth."""
        best_path = [None]
        best_len = [0]

        def dfs(u, path, visited):
            nonlocal best_path, best_len
            if time.time() > deadline:
                return
            if len(path) > max_depth:
                return
            if u == target and len(path) >= 2:
                if len(path) > best_len[0]:
                    best_len[0] = len(path)
                    best_path[0] = path[:]
                return
            for v in sorted(adj[u]):
                if v in visited:
                    continue
                if (u, v) == banned_edge or (v, u) == banned_edge:
                    continue
                if (u, v) not in alive_edges:
                    continue
                visited.add(v)
                path.append(v)
                dfs(v, path, visited)
                path.pop()
                visited.discard(v)

        visited = {start}
        dfs(start, [start], visited)
        return best_path[0]

    def find_best_cycle():
        if time.time() > deadline:
            return None

        best_cycle = None
        best_len = 0

        unique_edges = set()
        for u, v in alive_edges:
            if u < v:
                unique_edges.add((u, v))

        if not unique_edges:
            return None

        edge_list = sorted(unique_edges, key=lambda e: (len(adj[e[0]]) + len(adj[e[1]]), e), reverse=True)

        for edge in edge_list[:80]:
            if time.time() > deadline:
                break
            if edge not in unique_edges:
                continue
            path = find_path_avoiding_edge(edge[0], edge[1], edge)
            if path is not None:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cycle = path

        return best_cycle

    while True:
        if time.time() > deadline:
            break
        if not alive_edges:
            break

        cycle = find_best_cycle()
        if cycle is None:
            break

        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            alive_edges.discard((u, v))
            alive_edges.discard((v, u))
            adj[u].discard(v)
            adj[v].discard(u)

        pieces.append([int(v) for v in cycle])

    remaining = set()
    for u, v in alive_edges:
        if u < v:
            remaining.add((u, v))

    for u, v in sorted(remaining):
        pieces.append([u, v])

    return pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    start_time = time.time()
    results = []

    for i, (n, p, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        remaining_time = 145.0 - elapsed
        graphs_left = len(specs) - i
        alloc_time = remaining_time / graphs_left
        alloc_time = max(2.0, min(alloc_time, 45.0))

        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, time_limit=alloc_time)
        results.append(pieces)

    return results
