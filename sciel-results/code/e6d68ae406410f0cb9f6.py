"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    """Return three lists of pieces, one per fixed instance graph."""
    deadline = time.time() + 130  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, remaining, start, max_depth=15):
        """Find a long simple cycle starting and ending at start using DFS."""
        best_cycle = None
        best_len = 0

        def dfs(node, path, visited):
            nonlocal best_cycle, best_len
            if len(path) > max_depth:
                return
            for nb in adj[node]:
                e = (min(node, nb), max(node, nb))
                if e not in remaining:
                    continue
                if nb == start and len(path) >= 3:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = list(path)
                elif nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, path, visited)
                    path.pop()
                    visited.discard(nb)

        visited = {start}
        dfs(start, [start], visited)
        return best_cycle

    def decompose_graph(edges, time_budget=40):
        """Decompose a graph into minimum number of cycles and single edges."""
        if not edges:
            return []

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining = set(edges)
        pieces = []
        start_time = time.time()
        rng = random.Random(42)

        while remaining and (time.time() - start_time) < time_budget:
            # Try to find a long cycle
            best_cycle = None
            best_len = 0

            # Try from several random starting vertices
            candidates = list(remaining)
            rng.shuffle(candidates)
            tried_starts = set()

            for e in candidates[:20]:
                u, v = e
                for s in (u, v):
                    if s in tried_starts:
                        continue
                    tried_starts.add(s)
                    cycle = find_longest_cycle(adj, remaining, s, max_depth=12)
                    if cycle and len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle

            if best_cycle:
                # Extract cycle edges
                for i in range(len(best_cycle)):
                    a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                    rem = (min(a, b), max(a, b))
                    remaining.discard(rem)
                pieces.append(best_cycle)
            else:
                # No cycle found; take a single edge
                e = rng.choice(list(remaining))
                remaining.discard(e)
                pieces.append([e[0], e[1]])

        # Remaining edges as single pieces
        for e in remaining:
            pieces.append([e[0], e[1]])

        return pieces

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        if time.time() > deadline - 5:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        budget = min(40, max(5, deadline - time.time() - 5))
        pieces = decompose_graph(edges, time_budget=budget)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
