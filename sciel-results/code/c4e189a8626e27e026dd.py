"""Program that decomposes graphs into few cycles and edges."""
# EVOLVE-BLOCK-START
import time
from random import Random


def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n):
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))

        remaining = set(edge_set)
        pieces = []

        def find_longest_cycle_bfs():
            """Find a long cycle using a modified BFS/DFS approach."""
            best_cycle = None
            # Try starting from each vertex, preferring higher degree
            candidates = sorted(range(n), key=lambda x: -len([v for v in adj[x] if (min(x, v), max(x, v)) in remaining]))
            for start in candidates:
                if time.time() > deadline - 5:
                    break
                rem_deg = len([v for v in adj[start] if (min(start, v), max(start, v)) in remaining])
                if rem_deg < 2:
                    continue
                # DFS to find a cycle through start
                visited = {start}
                path = [start]
                found = None

                def dfs(node, depth):
                    nonlocal found
                    if found is not None:
                        return
                    if depth > 40:
                        return
                    for neighbor in sorted(adj[node], key=lambda x: -len([v for v in adj[x] if (min(x, v), max(x, v)) in remaining])):
                        e = (min(node, neighbor), max(node, neighbor))
                        if e not in remaining:
                            continue
                        if neighbor == start and depth >= 2:
                            found = path[:]
                            return
                        if neighbor not in visited:
                            visited.add(neighbor)
                            path.append(neighbor)
                            dfs(neighbor, depth + 1)
                            path.pop()
                            visited.discard(neighbor)
                            if found is not None:
                                return

                dfs(start, 0)
                if found is not None:
                    if best_cycle is None or len(found) > len(best_cycle):
                        best_cycle = found
                    if len(best_cycle) >= 15:
                        break
            return best_cycle

        def find_multiple_disjoint_cycles():
            """Try to find multiple disjoint cycles at once."""
            cycles = []
            temp_remaining = set(remaining)
            temp_adj = {i: set() for i in range(n)}
            for u, v in temp_remaining:
                temp_adj[u].add(v)
                temp_adj[v].add(u)

            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                found = False
                candidates = sorted(range(n), key=lambda x: -len(temp_adj[x]))
                for start in candidates:
                    if len(temp_adj[start]) < 2:
                        continue
                    visited = {start}
                    path = [start]
                    cycle = None

                    def dfs(node, depth):
                        nonlocal cycle
                        if cycle is not None:
                            return
                        if depth > 35:
                            return
                        for neighbor in sorted(temp_adj[node], key=lambda x: -len(temp_adj[x])):
                            e = (min(node, neighbor), max(node, neighbor))
                            if e not in temp_remaining:
                                continue
                            if neighbor == start and depth >= 2:
                                cycle = path[:]
                                return
                            if neighbor not in visited:
                                visited.add(neighbor)
                                path.append(neighbor)
                                dfs(neighbor, depth + 1)
                                path.pop()
                                visited.discard(neighbor)
                                if cycle is not None:
                                    return

                    dfs(start, 0)
                    if cycle is not None and len(cycle) >= 3:
                        cycles.append(cycle)
                        for i in range(len(cycle)):
                            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                            e = (min(u, v), max(u, v))
                            temp_remaining.discard(e)
                            temp_adj[u].discard(v)
                            temp_adj[v].discard(u)
                        found = True
                        break
                if not found:
                    break
            return cycles

        # Phase 1: Extract multiple disjoint cycles
        while time.time() < deadline - 30:
            if not remaining:
                break
            multi = find_multiple_disjoint_cycles()
            if not multi:
                break
            for cycle in multi:
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        remaining.discard(e)
                pieces.append(cycle)

        # Phase 2: Extract single longest cycles
        while time.time() < deadline - 15:
            if not remaining:
                break
            cycle = find_longest_cycle_bfs()
            if cycle is None:
                break
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
            pieces.append(cycle)

        # Phase 3: Try to merge remaining single edges into cycles
        # Build a graph of remaining edges and look for cycles
        rem_edges = list(remaining)
        if rem_edges:
            rem_adj = {i: set() for i in range(n)}
            rem_edge_set = set(remaining)
            for u, v in rem_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)

            # Find cycles in the remaining graph
            visited_global = set()
            while time.time() < deadline - 5:
                cycle_found = None
                for start in range(n):
                    if start in visited_global:
                        continue
                    if len(rem_adj[start]) < 2:
                        visited_global.add(start)
                        continue
                    visited = {start}
                    path = [start]

                    def dfs2(node, depth):
                        nonlocal cycle_found
                        if cycle_found is not None:
                            return
                        if depth > 30:
                            return
                        for neighbor in sorted(rem_adj[node]):
                            e = (min(node, neighbor), max(node, neighbor))
                            if e not in rem_edge_set:
                                continue
                            if neighbor == start and depth >= 2:
                                cycle_found = path[:]
                                return
                            if neighbor not in visited:
                                visited.add(neighbor)
                                path.append(neighbor)
                                dfs2(neighbor, depth + 1)
                                path.pop()
                                visited.discard(neighbor)
                                if cycle_found is not None:
                                    return

                    dfs2(start, 0)
                    if cycle_found is not None:
                        break
                    visited_global.add(start)

                if cycle_found is None:
                    break

                for i in range(len(cycle_found)):
                    u, v = cycle_found[i], cycle_found[(i + 1) % len(cycle_found)]
                    e = (min(u, v), max(u, v))
                    rem_edge_set.discard(e)
                    rem_adj[u].discard(v)
                    rem_adj[v].discard(u)
                pieces.append(cycle_found)

        # Remaining edges as singletons
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
