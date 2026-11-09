"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_path(start, adj, available_edges, n):
        """Find longest simple path starting from start using only available_edges."""
        best = [start]
        best_len = 0

        def dfs(node, path, used_edges):
            nonlocal best, best_len
            if len(path) - 1 > best_len:
                best_len = len(path) - 1
                best = list(path)

            for nb in adj[node]:
                e = tuple(sorted((node, nb)))
                if e in available_edges and e not in used_edges:
                    used_edges.add(e)
                    path.append(nb)
                    dfs(nb, path, used_edges)
                    path.pop()
                    used_edges.discard(e)

        dfs(start, [start], set())
        return best

    def path_cover(edges, n, time_limit):
        if not edges:
            return []

        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        remaining = set(edges)
        paths = []

        while remaining:
            if time.time() > time_limit:
                for e in remaining:
                    paths.append(list(e))
                break

            # Pick a random start vertex that has remaining edges
            candidates = set()
            for e in remaining:
                candidates.add(e[0])
                candidates.add(e[1])

            best_path = None
            best_len = -1

            # Try a few random starts to find a long path
            import random as _rnd
            _rnd.seed(hash(frozenset(remaining)) % (2**32))
            start_list = list(candidates)
            _rnd.shuffle(start_list)
            max_starts = min(len(start_list), 20)

            for start in start_list[:max_starts]:
                p = find_longest_path(start, adj, remaining, n)
                if len(p) - 1 > best_len:
                    best_len = len(p) - 1
                    best_path = p

            if best_path is None or len(best_path) < 2:
                for e in remaining:
                    paths.append(list(e))
                break

            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i + 1])))
                remaining.discard(e)

            paths.append(best_path)

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        n = spec[0]
        paths = path_cover(edges, n, deadline - 2)
        results.append(paths)

    return results
# EVOLVE-BLOCK-END
