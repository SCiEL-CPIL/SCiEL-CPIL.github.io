"""Path decomposition via edge-coverage greedy with odd-vertex bias."""
import time
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, time_limit=40.0):
        """Decompose edges into edge-disjoint paths covering all edges.
        Returns list of paths (each path is a list of vertices)."""
        if not edges:
            return []

        # Build adjacency and degree info
        adj = defaultdict(set)
        degree = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            degree[u] += 1
            degree[v] += 1

        # Track remaining edges
        remaining = set(edges)
        used_edges = set()

        def get_odd_vertices():
            return sorted([v for v in adj if degree[v] % 2 == 1])

        def try_extend(path, used_in_path):
            """Try to extend path by finding unused edges from the last vertex."""
            last = path[-1]
            # Get unused neighbors
            candidates = []
            for nb in adj[last]:
                e = tuple(sorted((last, nb)))
                if e not in used_edges:
                    # Check if nb is already in path (would create cycle)
                    if nb not in used_in_path:
                        candidates.append((nb, degree[nb] % 2 == 1, len(adj[nb] - used_in_path)))
            if not candidates:
                return False
            # Prefer odd-degree vertices, then lower remaining degree
            candidates.sort(key=lambda x: (not x[1], x[2]))
            best_nb = candidates[0][0]
            e = tuple(sorted((last, best_nb)))
            used_edges.add(e)
            remaining.discard(e)
            path.append(best_nb)
            used_in_path.add(best_nb)
            return True

        paths = []
        # Start from odd-degree vertices first (they must be endpoints)
        odd_verts = get_odd_vertices()
        all_verts = sorted(adj.keys())

        # Priority: odd vertices first, then others by degree
        start_priority = odd_verts + [v for v in all_verts if v not in set(odd_verts)]

        for start in start_priority:
            if time.time() - start_time > time_limit:
                break
            # Check if start has any unused edges
            has_unused = False
            for nb in adj[start]:
                e = tuple(sorted((start, nb)))
                if e not in used_edges:
                    has_unused = True
                    break
            if not has_unused:
                continue

            # Build path starting from start
            path = [start]
            used_in_path = {start}

            # Extend greedily
            while True:
                if time.time() - start_time > time_limit:
                    break
                if not try_extend(path, used_in_path):
                    break

            if len(path) > 1:
                paths.append(path)

        # Handle any remaining edges (shouldn't happen if graph is connected)
        if remaining:
            # Fallback: each remaining edge as its own path
            for e in sorted(remaining):
                paths.append(list(e))

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() - start_time > deadline - 5:
            # Not enough time, return what we have with trivial fallback
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        paths = decompose(edges, time_limit=deadline - (time.time() - start_time) - 5)
        results.append(paths)

    return results
