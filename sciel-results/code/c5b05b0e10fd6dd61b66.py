import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def greedy_walk(adj, remaining_edges, deg, start_v, rng, temperature, avoid_cycles):
        """
        Perform a single greedy walk from start_v.
        temperature: 0 = deterministic (pick min-degree neighbor), 1 = random
        avoid_cycles: if True, don't revisit vertices (simple path)
        """
        path = [start_v]
        visited = {start_v} if avoid_cycles else set()

        while True:
            v = path[-1]
            valid_neighbors = []
            for u in adj[v]:
                edge = (min(v, u), max(v, u))
                if edge in remaining_edges:
                    if avoid_cycles and u in visited:
                        continue
                    valid_neighbors.append(u)

            if not valid_neighbors:
                break

            if temperature <= 0.01:
                # Deterministic: pick neighbor with fewest remaining edges (degree)
                next_v = min(valid_neighbors, key=lambda u: deg.get(u, 0))
            else:
                # Weighted random: favor low-degree neighbors
                weights = []
                for u in valid_neighbors:
                    d = deg.get(u, 0)
                    weights.append(1.0 / (d + 1.0))
                total_w = sum(weights)
                if total_w == 0:
                    next_v = valid_neighbors[0]
                else:
                    r = rng.random() * total_w
                    cum = 0
                    next_v = valid_neighbors[-1]
                    for i, u in enumerate(valid_neighbors):
                        cum += weights[i]
                        if r <= cum:
                            next_v = u
                            break

            edge = (min(v, next_v), max(v, next_v))
            remaining_edges.discard(edge)
            deg[v] -= 1
            deg[next_v] -= 1
            path.append(next_v)
            if avoid_cycles:
                visited.add(next_v)

        return path

    def path_cover_single(edges, adj, rng, temperature, start_mode):
        """
        Run one full path-cover attempt with given parameters.
        Returns list of paths.
        """
        remaining = set(map(tuple, edges))
        deg = defaultdict(int)
        for e in remaining:
            deg[e[0]] += 1
            deg[e[1]] += 1

        paths = []
        while remaining:
            if time.time() > deadline:
                # Fallback: add remaining edges as single-edge paths
                for e in remaining:
                    paths.append(list(e))
                break

            # Choose start vertex
            if start_mode == 'leaf':
                leaves = [v for e in remaining for v in e if deg[v] == 1]
                if leaves:
                    start_v = rng.choice(leaves) if temperature > 0.01 else leaves[0]
                else:
                    min_deg = min(deg[v] for e in remaining for v in e)
                    candidates = [v for e in remaining for v in e if deg[v] == min_deg]
                    start_v = rng.choice(candidates) if temperature > 0.01 else candidates[0]
            elif start_mode == 'min_deg':
                min_deg = min(deg[v] for e in remaining for v in e)
                candidates = [v for e in remaining for v in e if deg[v] == min_deg]
                start_v = rng.choice(candidates) if temperature > 0.01 else candidates[0]
            else:  # random
                all_vertices = set()
                for e in remaining:
                    all_vertices.add(e[0])
                    all_vertices.add(e[1])
                start_v = rng.choice(list(all_vertices))

            path = greedy_walk(adj, remaining, deg, start_v, rng, temperature, True)
            if len(path) > 1:
                paths.append(path)
            else:
                # Single vertex path, try to extend
                for u in adj[start_v]:
                    edge = (min(start_v, u), max(start_v, u))
                    if edge in remaining:
                        path.append(u)
                        remaining.discard(edge)
                        deg[start_v] -= 1
                        deg[u] -= 1
                        break
                if len(path) > 1:
                    paths.append(path)
                else:
                    paths.append(path)

        # Handle any remaining edges
        for e in remaining:
            paths.append(list(e))

        return paths

    def merge_paths(paths, edges, adj):
        """
        Try to merge paths by finding connecting edges between endpoints.
        Returns improved list of paths.
        """
        if len(paths) < 2:
            return paths

        # Build set of all edges for quick lookup
        all_edges = set(map(tuple, edges))

        # Build set of edges used in current paths
        used_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                used_edges.add((min(u, v), max(u, v)))

        remaining_edges = all_edges - used_edges

        improved = True
        max_iters = 200
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break

            for i in range(len(paths)):
                if improved:
                    break
                for j in range(len(paths)):
                    if i == j:
                        continue

                    # Try connecting end of path i to start of path j
                    end_i = paths[i][-1]
                    start_j = paths[j][0]
                    edge = (min(end_i, start_j), max(end_i, start_j))
                    if edge in remaining_edges:
                        # Check that start_j is not already in path i
                        if start_j not in paths[i]:
                            paths[i] = paths[i] + paths[j]
                            remaining_edges.discard(edge)
                            paths.pop(j)
                            improved = True
                            break

                    # Try connecting start of path i to end of path j
                    start_i = paths[i][0]
                    end_j = paths[j][-1]
                    edge = (min(start_i, end_j), max(start_i, end_j))
                    if edge in remaining_edges:
                        # Check that end_j is not already in path i
                        if end_j not in paths[i]:
                            paths[i] = list(reversed(paths[j])) + paths[i]
                            remaining_edges.discard(edge)
                            paths.pop(j)
                            improved = True
                            break

        return paths

    def solve_graph(edges, n):
        """
        Solve path cover for a single graph using multiple strategies.
        Returns best list of paths found.
        """
        adj = build_adj(edges)
        best_cover = None
        best_count = len(edges)  # Upper bound: each edge as separate path

        # Define strategies: (temperature, start_mode)
        strategies = [
            (0.0, 'leaf'),
            (0.0, 'min_deg'),
            (0.2, 'leaf'),
            (0.4, 'leaf'),
            (0.6, 'leaf'),
            (0.8, 'leaf'),
            (1.0, 'leaf'),
            (0.3, 'min_deg'),
            (0.5, 'min_deg'),
            (0.7, 'min_deg'),
            (0.9, 'min_deg'),
            (0.5, 'random'),
            (0.8, 'random'),
        ]

        # Run multiple trials per strategy
        for temp, mode in strategies:
            for trial in range(40):
                if time.time() > deadline:
                    break

                rng = random.Random(hash((temp, mode, trial)) % (2**32))
                cover = path_cover_single(edges, adj, rng, temp, mode)
                cover = merge_paths(cover, edges, adj)
                count = len(cover)

                if count < best_count:
                    best_count = count
                    best_cover = cover

                    # Early termination if we hit a very good solution
                    if best_count <= 8:
                        break

            if time.time() > deadline:
                break
            if best_count <= 8:
                break

        if best_cover is None:
            best_cover = [list(e) for e in edges]

        return best_cover

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        n = spec[0]
        cover = solve_graph(edges, n)
        results.append(cover)

    return results
