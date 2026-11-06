import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_all_cycles_shortest_first(adj, n, remaining_edges, time_limit):
        """Find cycles greedily, shortest first, to maximize cycle count."""
        cycles = []
        adj_local = {i: set(adj[i]) for i in range(n)}
        used_edges = set()

        while len(used_edges) < len(remaining_edges) and time.time() < time_limit:
            best_cycle = None
            best_len = float('inf')

            # Try BFS from each vertex to find shortest cycle
            for start in range(n):
                if not adj_local[start]:
                    continue

                # BFS to find shortest cycle through start
                visited = {start: 0}
                parent = {start: None}
                queue = deque([(start, 0)])

                while queue:
                    u, dist = queue.popleft()
                    if dist + 1 >= best_len:
                        continue
                    for v in sorted(adj_local[u]):
                        e = (min(u, v), max(u, v))
                        if e in used_edges:
                            continue
                        if v == start:
                            if dist + 1 >= 3 and dist + 1 < best_len:
                                # Reconstruct cycle
                                cycle = [start]
                                cur = u
                                while cur != start:
                                    cur = parent[cur]
                                    cycle.append(cur)
                                best_cycle = cycle
                                best_len = dist + 1
                                break
                        elif v not in visited:
                            visited[v] = dist + 1
                            parent[v] = u
                            queue.append((v, dist + 1))
                    if best_cycle is not None:
                        break
                if best_cycle is not None:
                    break

            if best_cycle is None:
                break

            cycles.append(best_cycle)
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = (min(u, v), max(u, v))
                used_edges.add(e)
                adj_local[u].discard(v)
                adj_local[v].discard(u)

        return cycles

    def find_cycles_max_coverage(adj, n, remaining_edges, time_limit):
        """
        Find cycles to maximize coverage. Use a strategy that tries to find
        long cycles first (covering more edges per piece), but also ensures
        we don't leave short cycles unused.
        """
        cycles = []
        adj_local = {i: set(adj[i]) for i in range(n)}
        used_edges = set()

        # First pass: find longest cycles
        while len(used_edges) < len(remaining_edges) and time.time() < time_limit:
            best_cycle = None
            best_len = 0

            for start in range(n):
                if not adj_local[start]:
                    continue

                # DFS to find longest cycle through start
                visited = {start: 0}
                parent = {start: None}
                in_stack = {start}

                def dfs(u):
                    nonlocal best_cycle, best_len
                    for v in sorted(adj_local[u]):
                        e = (min(u, v), max(u, v))
                        if e in used_edges:
                            continue
                        if v not in visited:
                            visited[v] = visited[u] + 1
                            parent[v] = u
                            in_stack.add(v)
                            dfs(v)
                            in_stack.remove(v)
                        elif v in in_stack:
                            length = visited[u] - visited[v] + 1
                            if length >= 3 and length > best_len:
                                cycle = [u]
                                cur = u
                                while cur != v:
                                    cur = parent[cur]
                                    cycle.append(cur)
                                best_cycle = cycle
                                best_len = length

                dfs(start)

            if best_cycle is None:
                break

            cycles.append(best_cycle)
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = (min(u, v), max(u, v))
                used_edges.add(e)
                adj_local[u].discard(v)
                adj_local[v].discard(u)

        return cycles

    def find_cycles_by_degree(adj, n, remaining_edges, time_limit):
        """
        Strategy: prioritize vertices with high degree, find cycles through them.
        """
        cycles = []
        adj_local = {i: set(adj[i]) for i in range(n)}
        used_edges = set()

        # Sort vertices by degree (descending)
        vertices_by_degree = sorted(range(n), key=lambda x: len(adj_local[x]), reverse=True)

        while len(used_edges) < len(remaining_edges) and time.time() < time_limit:
            best_cycle = None
            best_score = 0

            for start in vertices_by_degree:
                if not adj_local[start]:
                    continue

                # Try to find a cycle through start
                visited = {start: 0}
                parent = {start: None}
                in_stack = {start}

                def dfs(u):
                    nonlocal best_cycle, best_score
                    for v in sorted(adj_local[u]):
                        e = (min(u, v), max(u, v))
                        if e in used_edges:
                            continue
                        if v not in visited:
                            visited[v] = visited[u] + 1
                            parent[v] = u
                            in_stack.add(v)
                            dfs(v)
                            in_stack.remove(v)
                        elif v in in_stack:
                            length = visited[u] - visited[v] + 1
                            if length >= 3:
                                # Score: prefer longer cycles
                                score = length
                                if score > best_score:
                                    cycle = [u]
                                    cur = u
                                    while cur != v:
                                        cur = parent[cur]
                                        cycle.append(cur)
                                    best_cycle = cycle
                                    best_score = score

                dfs(start)

            if best_cycle is None:
                break

            cycles.append(best_cycle)
            for i in range(len(best_cycle)):
                u = cycles[-1][i]
                v = cycles[-1][(i + 1) % len(cycles[-1])]
                e = (min(u, v), max(u, v))
                used_edges.add(e)
                adj_local[u].discard(v)
                adj_local[v].discard(u)

        return cycles

    def decompose_graph(edges, n, time_limit):
        """Decompose graph into minimum number of cycles and edges."""
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        pieces = []
        remaining_edges = set(edges)

        # Try multiple strategies and pick the best
        strategies = [
            ("shortest_first", find_all_cycles_shortest_first),
            ("max_coverage", find_cycles_max_coverage),
            ("by_degree", find_cycles_by_degree),
        ]

        best_pieces = None
        best_count = float('inf')

        for name, strategy in strategies:
            if time.time() > time_limit:
                break

            cycles = strategy(adj, n, remaining_edges, time_limit)

            # Count pieces: cycles + remaining single edges
            cycle_edges = set()
            for cycle in cycles:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    cycle_edges.add((min(u, v), max(u, v)))

            single_edges = remaining_edges - cycle_edges
            total_pieces = len(cycles) + len(single_edges)

            if total_pieces < best_count:
                best_count = total_pieces
                best_pieces = cycles + [list(e) for e in single_edges]

        # Fallback: if no cycles found, use single edges
        if best_pieces is None:
            best_pieces = [list(e) for e in remaining_edges]

        return best_pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        n = spec[0]

        # Allocate time per graph
        remaining_time = deadline - time.time()
        per_graph_time = max(remaining_time / max(1, len(specs) - specs.index(spec)), 10.0)

        pieces = decompose_graph(edges, n, time.time() + per_graph_time)
        results.append(pieces)

    return results
