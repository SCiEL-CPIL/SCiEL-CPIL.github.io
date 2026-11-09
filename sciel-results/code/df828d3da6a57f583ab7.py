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

    def bfs_distances(start, adj):
        """BFS from start, return dict of distances."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        return dist

    def decompose(edges):
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining_edges = set(edges)
        paths = []

        # Compute degrees
        degree = defaultdict(int)
        for u, v in remaining_edges:
            degree[u] += 1
            degree[v] += 1

        # Find odd-degree vertices
        odd_vertices = sorted([v for v in degree if degree[v] % 2 == 1])

        # Use minimum-weight matching to pair odd vertices optimally
        # The weight is the shortest path distance between them
        if len(odd_vertices) >= 2:
            m = len(odd_vertices)

            # Compute pairwise shortest path distances using BFS from each odd vertex
            dist_matrix = np.zeros((m, m))
            for i in range(m):
                dist_from_i = bfs_distances(odd_vertices[i], adj)
                for j in range(i + 1, m):
                    if odd_vertices[j] in dist_from_i:
                        d = dist_from_i[odd_vertices[j]]
                        dist_matrix[i, j] = d
                        dist_matrix[j, i] = d
                    else:
                        dist_matrix[i, j] = 1000
                        dist_matrix[j, i] = 1000

            # Find minimum weight perfect matching
            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            # Extract pairs
            pairs = []
            for r, c in zip(row_ind, col_ind):
                if r != c:
                    pairs.append((odd_vertices[r], odd_vertices[c]))

            # Remove duplicates (matching is symmetric)
            seen = set()
            unique_pairs = []
            for a, b in pairs:
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    unique_pairs.append((a, b))

            # For each pair, try to find a path
            for (a, b) in unique_pairs:
                if time.time() - start_time > time_limit - 5:
                    break
                # Find a simple path from a to b using remaining edges
                path = _find_simple_path(a, b, remaining_edges, adj)
                if path and len(path) >= 2:
                    paths.append(path)
                    for i in range(len(path) - 1):
                        e = tuple(sorted((path[i], path[i + 1])))
                        remaining_edges.discard(e)
                        degree[path[i]] -= 1
                        degree[path[i + 1]] -= 1

        # Greedy: cover remaining edges
        while remaining_edges:
            if time.time() - start_time > time_limit - 2:
                # Emergency: add remaining edges as individual paths
                for e in remaining_edges:
                    paths.append(list(e))
                break

            # Recompute degrees for remaining edges
            rem_degree = defaultdict(int)
            for u, v in remaining_edges:
                rem_degree[u] += 1
                rem_degree[v] += 1

            # Pick an edge to start a path
            # Choose edge where sum of endpoint degrees is highest
            best_edge = None
            best_key = -1
            for e in remaining_edges:
                u, v = e
                k = rem_degree[u] + rem_degree[v]
                if k > best_key:
                    best_key = k
                    best_edge = e

            if best_edge is None:
                break

            u, v = best_edge
            path = [u, v]
            remaining_edges.discard(best_edge)
            path_set = set(path)

            # Extend from u end
            while True:
                current = path[0]
                candidates = [nb for nb in adj[current]
                              if tuple(sorted((current, nb))) in remaining_edges
                              and nb not in path_set]
                if not candidates:
                    break
                candidates.sort(key=lambda nb: -rem_degree[nb])
                next_node = candidates[0]
                path.insert(0, next_node)
                path_set.add(next_node)
                remaining_edges.discard(tuple(sorted((current, next_node))))
                rem_degree[current] -= 1
                rem_degree[next_node] -= 1

            # Extend from v end
            while True:
                current = path[-1]
                candidates = [nb for nb in adj[current]
                              if tuple(sorted((current, nb))) in remaining_edges
                              and nb not in path_set]
                if not candidates:
                    break
                candidates.sort(key=lambda nb: -rem_degree[nb])
                next_node = candidates[0]
                path.append(next_node)
                path_set.add(next_node)
                remaining_edges.discard(tuple(sorted((current, next_node))))
                rem_degree[current] -= 1
                rem_degree[next_node] -= 1

            paths.append(path)

        # Local search: try to merge paths by finding shared edges
        # This is a heuristic improvement step
        paths = _local_search_merge(paths, adj, remaining_edges, start_time, time_limit)

        return paths

    def _find_simple_path(start, end, remaining_edges, adj, max_depth=60):
        """Find a simple path from start to end using only edges in remaining_edges."""
        if start == end:
            return [start]

        visited = {start}
        path = [start]

        def dfs(node, path, visited):
            if node == end:
                return list(path)
            if len(path) > max_depth:
                return None

            neighbors = list(adj[node])
            # Sort neighbors: prefer those with fewer remaining options
            def neighbor_score(nb):
                count = 0
                for nb2 in adj[nb]:
                    e = tuple(sorted((nb, nb2)))
                    if e in remaining_edges and nb2 not in visited:
                        count += 1
                return count

            neighbors.sort(key=neighbor_score)

            for nb in neighbors:
                if nb in visited:
                    continue
                e = tuple(sorted((node, nb)))
                if e not in remaining_edges:
                    continue

                # Try this edge
                remaining_edges.discard(e)
                visited.add(nb)
                path.append(nb)

                result = dfs(nb, path, visited)
                if result is not None:
                    return result

                # Backtrack
                path.pop()
                visited.discard(nb)
                remaining_edges.add(e)

            return None

        return dfs(start, path, visited)

    def _local_search_merge(paths, adj, remaining_edges, start_time, time_limit):
        """Try to merge adjacent paths by finding shared edges."""
        # This is a simplified local search that tries to merge paths
        # by checking if the end of one path connects to the start of another
        changed = True
        while changed and (time.time() - start_time) < (time_limit - 3):
            changed = False
            new_paths = list(paths)
            for i in range(len(new_paths)):
                for j in range(i + 1, len(new_paths)):
                    # Check if we can merge path i and path j
                    # Try: end of i -> start of j
                    end_i = new_paths[i][-1]
                    start_j = new_paths[j][0]
                    e = tuple(sorted((end_i, start_j)))
                    if e in remaining_edges:
                        # Merge: extend path i with path j
                        merged = new_paths[i] + new_paths[j][1:]
                        new_paths[i] = merged
                        new_paths.pop(j)
                        remaining_edges.discard(e)
                        changed = True
                        break
                    # Try: end of j -> start of i
                    end_j = new_paths[j][-1]
                    start_i = new_paths[i][0]
                    e = tuple(sorted((end_j, start_i)))
                    if e in remaining_edges:
                        # Merge: extend path j with path i
                        merged = new_paths[j] + new_paths[i][1:]
                        new_paths[j] = merged
                        new_paths.pop(i)
                        remaining_edges.discard(e)
                        changed = True
                        break
                    # Try: start of i -> end of j
                    start_i = new_paths[i][0]
                    end_j = new_paths[j][-1]
                    e = tuple(sorted((start_i, end_j)))
                    if e in remaining_edges:
                        # Merge: prepend path j to path i
                        merged = new_paths[j][::-1][1:] + new_paths[i]
                        new_paths[i] = merged
                        new_paths.pop(j)
                        remaining_edges.discard(e)
                        changed = True
                        break
                    # Try: start of j -> end of i
                    start_j = new_paths[j][0]
                    end_i = new_paths[i][-1]
                    e = tuple(sorted((start_j, end_i)))
                    if e in remaining_edges:
                        # Merge: prepend path i to path j
                        merged = new_paths[i][::-1][1:] + new_paths[j]
                        new_paths[j] = merged
                        new_paths.pop(i)
                        remaining_edges.discard(e)
                        changed = True
                        break
                if changed:
                    break
            paths = new_paths
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)

    return results
