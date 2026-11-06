"""Program that decomposes graphs into few cycles and edges."""
# EVOLVE-BLOCK-START
import time
from random import Random


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n):
        # Build adjacency
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))

        remaining = set(edge_set)
        pieces = []

        def find_cycle():
            """Find a cycle in the remaining graph using DFS."""
            visited = set()
            parent = {}
            cycle = None

            def dfs(node, start):
                nonlocal cycle
                if cycle is not None:
                    return
                visited.add(node)
                for neighbor in sorted(adj[node]):
                    if (min(node, neighbor), max(node, neighbor)) not in remaining:
                        continue
                    if neighbor == start and len(visited) >= 3:
                        # Found a cycle
                        cycle = [start]
                        cur = node
                        while cur != start:
                            cycle.append(cur)
                            cur = parent[cur]
                        return
                    if neighbor not in visited:
                        parent[neighbor] = node
                        dfs(neighbor, start)
                        if cycle is not None:
                            return

            for start in range(n):
                if len(adj[start]) < 2:
                    continue
                visited = set()
                parent = {}
                cycle = None
                dfs(start, start)
                if cycle is not None:
                    return cycle
            return None

        def find_longest_cycle():
            """Find a longer cycle by trying multiple starts."""
            best = None
            for start in range(n):
                if len(adj[start]) < 2:
                    continue
                visited = set()
                parent = {}
                cycle = None

                def dfs(node, start_node, depth):
                    nonlocal cycle, best
                    if cycle is not None:
                        return
                    if depth > 50:  # limit depth for speed
                        return
                    visited.add(node)
                    for neighbor in sorted(adj[node]):
                        if (min(node, neighbor), max(node, neighbor)) not in remaining:
                            continue
                        if neighbor == start_node and len(visited) >= 3:
                            new_cycle = [start_node]
                            cur = node
                            while cur != start_node:
                                new_cycle.append(cur)
                                cur = parent[cur]
                            if best is None or len(new_cycle) > len(best):
                                best = new_cycle
                            cycle = new_cycle
                            return
                        if neighbor not in visited:
                            parent[neighbor] = node
                            dfs(neighbor, start_node, depth + 1)
                            if cycle is not None:
                                return

                dfs(start, start, 0)
                if cycle is not None:
                    return cycle
            return best

        # Greedily extract cycles
        max_iterations = 500
        iter_count = 0
        while iter_count < max_iterations:
            iter_count += 1
            if time.time() > deadline - 10:
                break

            cycle = find_longest_cycle()
            if cycle is None:
                cycle = find_cycle()
            if cycle is None:
                break

            # Remove cycle edges from remaining
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)

            pieces.append(cycle)

        # Remaining edges become individual pieces
        for u, v in sorted(remaining):
            pieces.append([u, v])

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
