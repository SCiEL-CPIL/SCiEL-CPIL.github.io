"""Path decompositions of three fixed connected graphs (Gallai's conjecture)."""

# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = 120  # seconds, with safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def path_decompose(edges, n):
        """Decompose a connected graph into edge-disjoint paths covering all edges.
        Uses a greedy approach: repeatedly find a long path in the remaining graph."""
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        remaining_edges = set(edges)
        paths = []

        def find_longest_path(start, visited_edges):
            """Find a long path starting from 'start' using DFS with edge constraints."""
            best_path = [start]
            best_len = 0

            def dfs(node, path, used_edges):
                nonlocal best_path, best_len
                if len(path) > best_len:
                    best_path = list(path)
                    best_len = len(path)

                # Try to extend
                for nbr in adj[node]:
                    edge = tuple(sorted((node, nbr)))
                    if edge not in used_edges and edge not in visited_edges:
                        path.append(nbr)
                        used_edges.add(edge)
                        dfs(nbr, path, used_edges)
                        path.pop()
                        used_edges.remove(edge)

            dfs(start, [start], set())
            return best_path

        # Repeatedly find a path covering many edges
        while remaining_edges:
            # Pick a vertex with highest remaining degree as start
            deg = {}
            for e in remaining_edges:
                u, v = e
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1

            if not deg:
                break

            # Start from vertex with max degree
            start = max(deg.keys(), key=lambda x: deg[x])

            # DFS to find a long path
            path = [start]
            visited_edges = set()
            visited_nodes = {start}

            def extend_path(current):
                # Try neighbors with most remaining edges
                candidates = []
                for nbr in adj[current]:
                    edge = tuple(sorted((current, nbr)))
                    if edge in remaining_edges and edge not in visited_edges:
                        # Count how many edges nbr still has
                        nbr_deg = sum(1 for e in remaining_edges if e[0] == nbr or e[1] == nbr)
                        candidates.append((nbr_deg, nbr))
                candidates.sort(reverse=True)

                for _, nbr in candidates:
                    edge = tuple(sorted((current, nbr)))
                    if edge not in remaining_edges or edge in visited_edges:
                        continue
                    path.append(nbr)
                    visited_edges.add(edge)
                    remaining_edges.discard(edge)

                    # Check if we can continue
                    has_more = False
                    for nbr2 in adj[nbr]:
                        e2 = tuple(sorted((nbr, nbr2)))
                        if e2 in remaining_edges and e2 not in visited_edges:
                            has_more = True
                            break

                    if has_more:
                        extend_path(nbr)
                        return True
                    else:
                        # Dead end, backtrack one step
                        path.pop()
                        visited_edges.discard(edge)
                        remaining_edges.add(edge)
                        return False

                return False

            extend_path(start)

            if len(path) >= 2:
                paths.append(path)
            else:
                # Fallback: take any remaining edge
                e = next(iter(remaining_edges))
                paths.append([e[0], e[1]])
                remaining_edges.discard(e)

            # Check time
            if time.time() - start_time > deadline - 5:
                break

        # If there are still remaining edges (due to time limit or issues), add them as single-edge paths
        for e in remaining_edges:
            paths.append([e[0], e[1]])

        return paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []

    for spec in specs:
        if time.time() - start_time > deadline - 10:
            # Not enough time, return trivial solution for remaining
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue

        edges = instance(*spec)
        n = spec[0]
        paths = path_decompose(edges, n)
        results.append(paths)

    return results
# EVOLVE-BLOCK-END
