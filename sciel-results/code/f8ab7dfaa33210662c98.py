"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    # Deadline guard: 140s wall-clock safety margin
    t_start = time.time()
    DEADLINE = 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    def decompose_graph(edges, time_budget):
        """Decompose a graph into edge-disjoint paths covering all edges.
        
        Uses a randomized greedy approach with Warnsdorff-like heuristic:
        repeatedly pick a starting vertex and greedily extend a path,
        choosing the next vertex that leads to the fewest remaining options
        (to leave more flexible vertices for later).
        """
        if not edges:
            return []

        # Build adjacency and degree info
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # We'll work with a mutable set of remaining edges
        remaining = set(edges)
        paths = []

        # Helper: find a path starting from a vertex using greedy + backtracking-lite
        def find_path(start, remaining_set, adj_map):
            """Greedily extend a path from start, using Warnsdorff heuristic."""
            path = [start]
            used = set()
            current = start
            
            while True:
                # Find neighbors of current that have an unused edge
                candidates = []
                for nb in adj_map[current]:
                    edge = tuple(sorted((current, nb)))
                    if edge in remaining_set and edge not in used:
                        # Count how many unused edges nb has (excluding the one to current)
                        deg = 0
                        for nn in adj_map[nb]:
                            e2 = tuple(sorted((nb, nn)))
                            if e2 in remaining_set and e2 != edge:
                                deg += 1
                        candidates.append((deg, nb))
                
                if not candidates:
                    break
                
                # Warnsdorff: pick the neighbor with fewest remaining options
                candidates.sort()
                next_v = candidates[0][1]
                edge = tuple(sorted((current, next_v)))
                used.add(edge)
                remaining_set.discard(edge)
                path.append(next_v)
                current = next_v
            
            return path

        # Strategy: multiple restarts, each time greedily decompose
        best_paths = None
        best_count = len(edges)  # trivial upper bound: one path per edge

        # Number of restarts based on graph size and time budget
        n_vertices = max(max(u, v) for u, v in edges) + 1 if edges else 0
        max_restarts = 50 if n_vertices <= 25 else 30

        for restart in range(max_restarts):
            if time.time() - t_start > time_budget:
                break

            remaining_set = set(edges)
            paths_this = []

            # Shuffle the order in which we start paths
            # Prioritize vertices with odd degree (path endpoints in optimal decomp)
            deg = defaultdict(int)
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1

            # Start from vertices, preferring those with higher degree first
            # so we consume more edges per path
            vertices = list(range(n_vertices))
            random.shuffle(vertices)

            while remaining_set:
                # Pick a start vertex: prefer one that's an endpoint of a remaining edge
                # and has high degree
                start_candidates = []
                for v in vertices:
                    for nb in adj[v]:
                        e = tuple(sorted((v, nb)))
                        if e in remaining_set:
                            start_candidates.append(v)
                            break
                
                if not start_candidates:
                    break

                # Choose the candidate with highest remaining degree
                def rem_deg(v):
                    count = 0
                    for nb in adj[v]:
                        e = tuple(sorted((v, nb)))
                        if e in remaining_set:
                            count += 1
                    return count

                start_candidates.sort(key=rem_deg, reverse=True)
                start = start_candidates[0]

                path = find_path(start, remaining_set, adj)
                if len(path) > 1:
                    paths_this.append(path)
                else:
                    # Single vertex, shouldn't happen if there are edges, but safety
                    break

            if paths_this and len(paths_this) < best_count:
                best_count = len(paths_this)
                best_paths = [list(p) for p in paths_this]

            # Check if we've hit the theoretical lower bound
            odd_deg = sum(1 for v in vertices if deg[v] % 2 == 1)
            lower_bound = max(odd_deg // 2, 1) if edges else 0
            if best_count == lower_bound:
                break

        if best_paths is None:
            # Fallback: each edge as its own path
            best_paths = [[u, v] for u, v in edges]

        return best_paths

    results = []
    # Allocate time proportionally, but ensure each graph gets at least some time
    total_time_budget = DEADLINE - (time.time() - t_start) - 5.0  # 5s safety margin
    if total_time_budget < 10:
        total_time_budget = 5.0

    # Give more time to larger graphs
    weights = [n for n, p, s in specs]
    total_w = sum(weights)
    for i, (n, p, s) in enumerate(specs):
        graph_time = total_time_budget * weights[i] / total_w
        if graph_time < 2.0:
            graph_time = 2.0
        edges = instance(n, p, s)
        paths = decompose_graph(edges, graph_time)
        results.append(paths)
        # Update remaining time
        total_time_budget = DEADLINE - (time.time() - t_start) - 5.0
        if total_time_budget < 0:
            total_time_budget = 0

    return results
# EVOLVE-BLOCK-END
