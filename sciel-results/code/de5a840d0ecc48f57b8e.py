import random
import time
from collections import defaultdict, deque
import numpy as np
from scipy.optimize import linear_sum_assignment


def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def greedy_trail_decompose(edges):
        """Decompose edges into edge-disjoint trails (walks) greedily."""
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        remaining = set(edges)
        trails = []

        while remaining:
            # Pick an edge to start
            # Prefer edges incident to odd-degree vertices in the remaining subgraph
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1

            # Find odd-degree vertices in remaining
            odd = [v for v in deg if deg[v] % 2 == 1]

            # Try to start from an odd-degree vertex
            start_edge = None
            if odd:
                for v in odd:
                    for nb in adj[v]:
                        e = tuple(sorted((v, nb)))
                        if e in remaining:
                            start_edge = (v, nb)
                            break
                    if start_edge:
                        break

            if start_edge is None:
                start_edge = next(iter(remaining))

            # Build trail starting from start_edge
            u0, v0 = start_edge
            remaining.discard(tuple(sorted((u0, v0))))
            trail = [u0, v0]
            path_set = set(trail)

            # Extend from both ends
            while True:
                # Try to extend from the end (v0 side)
                extended = False
                current = trail[-1]
                candidates = [nb for nb in adj[current]
                              if tuple(sorted((current, nb))) in remaining
                              and nb not in path_set]
                if not candidates:
                    # Try from the start side
                    current = trail[0]
                    candidates = [nb for nb in adj[current]
                                  if tuple(sorted((current, nb))) in remaining
                                  and nb not in path_set]
                    if candidates:
                        # Pick candidate with fewest remaining options (greedy)
                        candidates.sort(key=lambda nb: sum(
                            1 for nb2 in adj[nb]
                            if tuple(sorted((nb, nb2))) in remaining))
                        next_node = candidates[0]
                        e = tuple(sorted((current, next_node)))
                        remaining.discard(e)
                        trail.insert(0, next_node)
                        path_set.add(next_node)
                        extended = True
                    else:
                        break
                else:
                    candidates.sort(key=lambda nb: sum(
                        1 for nb2 in adj[nb]
                        if tuple(sorted((nb, nb2))) in remaining))
                    next_node = candidates[0]
                    e = tuple(sorted((current, next_node)))
                    remaining.discard(e)
                    trail.append(next_node)
                    path_set.add(next_node)
                    extended = True

            trails.append(trail)

        return trails

    def split_trail(trail):
        """Split a trail with repeated vertices into simple paths."""
        if len(trail) <= 2:
            return [trail]

        # Find first repeated vertex
        seen = {}
        split_points = []
        for i, v in enumerate(trail):
            if v in seen:
                split_points.append((seen[v], i))
                break
            seen[v] = i

        if not split_points:
            return [trail]

        # Recursively split
        first_split = split_points[0]
        left = trail[:first_split[1] + 1]
        right = trail[first_split[0]:]

        result = []
        result.extend(split_trail(left))
        result.extend(split_trail(right))
        return result

    def local_search_merge(paths, edges_set, time_limit, start_time):
        """Iteratively try to merge paths by finding connecting edges."""
        adj = defaultdict(set)
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)

        changed = True
        max_iters = 200
        iter_count = 0

        while changed and iter_count < max_iters:
            iter_count += 1
            if time.time() - start_time > time_limit - 3:
                break
            changed = False
            n = len(paths)
            for i in range(n):
                if changed:
                    break
                for j in range(i + 1, n):
                    # Try all 4 orientations of merging path i and path j
                    merged = None

                    # end_i -> start_j
                    e = tuple(sorted((paths[i][-1], paths[j][0])))
                    if e in edges_set and len(set(paths[i]) & set(paths[j])) == 0:
                        merged = paths[i] + paths[j][1:]
                        edges_set.discard(e)

                    # start_i -> end_j
                    if merged is None:
                        e = tuple(sorted((paths[i][0], paths[j][-1])))
                        if e in edges_set and len(set(paths[i]) & set(paths[j])) == 0:
                            merged = paths[j][::-1][1:] + paths[i]
                            edges_set.discard(e)

                    # end_j -> start_i
                    if merged is None:
                        e = tuple(sorted((paths[j][-1], paths[i][0])))
                        if e in edges_set and len(set(paths[i]) & set(paths[j])) == 0:
                            merged = paths[j] + paths[i][1:]
                            edges_set.discard(e)

                    # start_j -> end_i
                    if merged is None:
                        e = tuple(sorted((paths[j][0], paths[i][-1])))
                        if e in edges_set and len(set(paths[i]) & set(paths[j])) == 0:
                            merged = paths[i][::-1][1:] + paths[j]
                            edges_set.discard(e)

                    if merged is not None:
                        paths[i] = merged
                        paths.pop(j)
                        changed = True
                        break

        return paths

    def decompose(edges):
        if not edges:
            return []

        edges_set = set(edges)

        # Step 1: Greedy trail decomposition
        trails = greedy_trail_decompose(edges)

        # Step 2: Split non-simple trails into simple paths
        paths = []
        for trail in trails:
            simple_paths = split_trail(trail)
            paths.extend(simple_paths)

        # Step 3: Local search to merge paths
        # Use a copy of the full edge set for the merge heuristic
        full_edges = set(edges)
        paths = local_search_merge(paths, full_edges, time_limit, start_time)

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)

    return results
