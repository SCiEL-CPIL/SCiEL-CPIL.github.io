"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_decomposition(edges, n, deadline):
        """Find decomposition with minimum pieces using multiple strategies."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []

        best_pieces = None
        best_count = len(edge_set) + 1

        # Strategy 1: Greedy with longest-cycle-first approach
        def greedy_longest_first():
            remaining = set(edge_set)
            pieces = []

            while remaining:
                # Build adjacency for remaining edges
                adj = defaultdict(set)
                for u, v in remaining:
                    adj[u].add(v)
                    adj[v].add(u)

                # Find the longest cycle in the remaining graph
                longest_cycle = None
                best_len = 2

                # Try from each vertex as start
                for start in sorted(adj.keys()):
                    if len(adj[start]) < 2:
                        continue

                    # Try to find a cycle starting from 'start'
                    cycle = _find_longest_cycle(start, adj, remaining, max_depth=min(30, len(remaining)))
                    if cycle and len(cycle) > best_len:
                        best_len = len(cycle)
                        longest_cycle = cycle

                        # If we found a very long cycle, we can stop early
                        if best_len >= len(remaining):
                            break

                if longest_cycle and len(longest_cycle) >= 3:
                    # Remove cycle edges from remaining
                    for i in range(len(longest_cycle)):
                        u, v = longest_cycle[i], longest_cycle[(i + 1) % len(longest_cycle)]
                        e = tuple(sorted((u, v)))
                        remaining.discard(e)
                    pieces.append(longest_cycle)
                else:
                    # No cycle found; take one edge as a piece
                    e = tuple(sorted(random.choice(list(remaining))))
                    remaining.discard(e)
                    pieces.append(list(e))

            return pieces

        def _find_longest_cycle(start, adj, remaining, max_depth=30):
            """Find the longest cycle starting at 'start' using DFS."""
            best_cycle = None
            best_len = 2

            def dfs(current, path, visited):
                nonlocal best_cycle, best_len

                if len(path) > best_len:
                    # Check if we can close the cycle
                    if current == start and len(path) >= 3:
                        best_cycle = path[:]
                        best_len = len(path)
                        return

                if len(path) >= max_depth:
                    return

                for nb in adj[current]:
                    e = tuple(sorted((current, nb)))
                    if e not in remaining:
                        continue

                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        if len(path) > best_len:
                            best_cycle = path[:]
                            best_len = len(path)
                    elif nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        dfs(nb, path, visited)
                        path.pop()
                        visited.discard(nb)

            dfs(start, [start], {start})
            return best_cycle

        # Strategy 2: Iterative improvement - start with greedy, then try to merge
        def greedy_with_improvement():
            remaining = set(edge_set)
            pieces = []

            while remaining:
                adj = defaultdict(set)
                for u, v in remaining:
                    adj[u].add(v)
                    adj[v].add(u)

                # Find a cycle using randomized walk
                cycle = None
                for attempt in range(20):
                    # Pick a random start vertex with degree >= 2
                    candidates = [v for v in adj if len(adj[v]) >= 2]
                    if not candidates:
                        break
                    start = random.choice(candidates)

                    # Random walk to find a cycle
                    path = [start]
                    visited = {start}
                    current = start
                    found = False

                    for _ in range(50):
                        neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining]
                        if not neighbors:
                            break

                        unvisited = [nb for nb in neighbors if nb not in visited]
                        if start in neighbors and len(path) >= 3:
                            # Can close cycle
                            found = True
                            break
                        elif unvisited:
                            next_node = random.choice(unvisited)
                        else:
                            break

                        path.append(next_node)
                        visited.add(next_node)
                        current = next_node

                    if found and len(path) >= 3:
                        if cycle is None or len(path) > len(cycle):
                            cycle = path[:]

                if cycle and len(cycle) >= 3:
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        remaining.discard(e)
                    pieces.append(cycle)
                else:
                    e = tuple(sorted(random.choice(list(remaining))))
                    remaining.discard(e)
                    pieces.append(list(e))

            return pieces

        # Try multiple strategies and keep the best
        strategies = [greedy_longest_first, greedy_with_improvement]
        for strategy in strategies:
            if time.time() > deadline - 10:
                break
            pieces = strategy()
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [p[:] for p in pieces]

        # If we still have time, try more iterations of randomized greedy
        for iteration in range(50):
            if time.time() > deadline - 10:
                break

            remaining = set(edge_set)
            pieces = []

            while remaining:
                adj = defaultdict(set)
                for u, v in remaining:
                    adj[u].add(v)
                    adj[v].add(u)

                # Find a cycle using randomized approach
                cycle = None
                best_cycle_len = 2

                for attempt in range(30):
                    candidates = [v for v in adj if len(adj[v]) >= 2]
                    if not candidates:
                        break
                    start = random.choice(candidates)

                    path = [start]
                    visited = {start}
                    current = start
                    found = False

                    for _ in range(40):
                        neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining]
                        if not neighbors:
                            break

                        unvisited = [nb for nb in neighbors if nb not in visited]
                        if start in neighbors and len(path) >= 3:
                            found = True
                            break
                        elif unvisited:
                            # Prefer neighbors with higher degree
                            unvisited.sort(key=lambda x: len(adj[x]), reverse=True)
                            if random.random() < 0.4:
                                next_node = random.choice(unvisited)
                            else:
                                next_node = unvisited[0]
                        else:
                            break

                        path.append(next_node)
                        visited.add(next_node)
                        current = next_node

                    if found and len(path) >= 3 and len(path) > best_cycle_len:
                        best_cycle_len = len(path)
                        cycle = path[:]

                if cycle and len(cycle) >= 3:
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        remaining.discard(e)
                    pieces.append(cycle)
                else:
                    e = tuple(sorted(random.choice(list(remaining))))
                    remaining.discard(e)
                    pieces.append(list(e))

            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [p[:] for p in pieces]

        return best_pieces if best_pieces else [list(e) for e in edge_set]

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = find_decomposition(edges, n, deadline)
        results.append(pieces)
        if time.time() > deadline - 5:
            # Fallback: just use individual edges for remaining graphs
            for spec2 in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results):]:
                n2, p2, seed2 = spec2
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break

    return results
# EVOLVE-BLOCK-END
