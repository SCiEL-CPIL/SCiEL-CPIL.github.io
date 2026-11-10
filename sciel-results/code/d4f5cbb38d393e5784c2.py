"""Decompose three fixed Erdos-Renyi-ish graphs into few cycles + edges.

Strategy:
  * Minimizing pieces == maximizing number of edge-disjoint cycles, because
    any two vertex-disjoint cycles can be merged into one cycle (using a
    leftover edge as the connector), so total pieces = (#cycles) + (#singles).
  * Greedy longest-cycle extraction is optimal for the cycle count: a longer
    cycle frees more vertices to form additional cycles and is never worse
    than single edges.
  * Find longest cycles via randomized DFS (restarts, low-degree bias) with a
    hard wall-clock deadline; leftover edges become single-edge pieces.
  * Merge all cycles into one chain (connectors are leftover edges).
"""

import time
from collections import defaultdict
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_best_decomposition(edges, n):
        """Find decomposition with minimum pieces using randomized greedy cycle extraction."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []

        best_pieces = None
        best_count = len(edge_set) + 1

        # Use 2-core decomposition to restrict to cycle-supporting subgraphs
        def get_2_core(edge_list):
            """Extract the 2-core of the graph (vertices with degree >= 2 iteratively)."""
            if not edge_list:
                return set(), set()
            adj = defaultdict(set)
            for u, v in edge_list:
                adj[u].add(v)
                adj[v].add(u)

            # Iteratively remove vertices with degree < 2
            changed = True
            while changed:
                changed = False
                to_remove = set()
                for u in list(adj.keys()):
                    if len(adj[u]) < 2:
                        to_remove.add(u)
                if to_remove:
                    changed = True
                    for u in to_remove:
                        for v in adj[u]:
                            adj[v].discard(u)
                        del adj[u]

            # Build edge set from 2-core
            core_edges = set()
            for u in adj:
                for v in adj[u]:
                    if u < v:
                        core_edges.add((u, v))
            return core_edges, adj

        # For each restart, try different strategies
        for iteration in range(300):
            if time.time() > deadline - 10:
                break

            # Get 2-core to focus on cycle-supporting regions
            core_edges, core_adj = get_2_core(edge_set)

            remaining = set(edge_set)
            pieces = []

            # Phase 1: Extract cycles from the 2-core region
            # Use randomized greedy with bias toward longer cycles
            remaining_core = set(core_edges)
            while remaining_core:
                if time.time() > deadline - 15:
                    break

                # Pick vertex with highest degree in remaining_core
                max_deg = -1
                start = None
                for u in core_adj:
                    deg = len([nb for nb in core_adj[u] if (min(u, nb), max(u, nb)) in remaining_core])
                    if deg > max_deg:
                        max_deg = deg
                        start = u

                if start is None or max_deg < 2:
                    # No cycles possible in remaining core
                    break

                # Try to find a cycle using randomized DFS
                cycle = _find_cycle_in_subgraph(start, core_adj, remaining_core)

                if cycle and len(cycle) >= 3:
                    # Remove cycle edges
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        remaining_core.discard(e)
                        remaining.discard(e)
                    pieces.append(cycle)
                else:
                    # No cycle found, break out of core phase
                    break

            # Phase 2: Handle remaining edges (including non-core edges)
            # Use greedy edge extraction or short cycles
            while remaining:
                if time.time() > deadline - 15:
                    break

                # Try to find any cycle in remaining
                # Pick a random edge and try to extend it
                sample_edges = list(remaining)
                random.shuffle(sample_edges)

                found_cycle = False
                for e in sample_edges[:5]:  # Try up to 5 random edges
                    u, v = e
                    # Try to find a path from v back to u avoiding the edge (u,v)
                    path = _find_path_avoiding_edge(u, v, core_adj, remaining, exclude_edge=e)
                    if path and len(path) >= 2:
                        cycle = [u] + path
                        if len(cycle) >= 3:
                            # Remove cycle edges
                            for i in range(len(cycle)):
                                cu, cv = cycle[i], cycle[(i + 1) % len(cycle)]
                                ce = tuple(sorted((cu, cv)))
                                remaining.discard(ce)
                            pieces.append(cycle)
                            found_cycle = True
                            break

                if not found_cycle:
                    # Just take one edge
                    e = tuple(sorted(random.choice(list(remaining))))
                    remaining.discard(e)
                    pieces.append(list(e))

            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [p[:] for p in pieces]

        return best_pieces if best_pieces else [list(e) for e in edge_set]

    def _find_cycle_in_subgraph(start, adj, remaining_edges):
        """Find a cycle starting at 'start' using randomized DFS."""
        best_cycle = None

        for attempt in range(50):
            path = [start]
            visited = {start}
            current = start
            found = False

            for _ in range(100):
                neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining_edges]
                if not neighbors:
                    break

                unvisited = [nb for nb in neighbors if nb not in visited]

                if unvisited:
                    # Prefer higher degree neighbors
                    unvisited.sort(key=lambda x: len(adj[x]), reverse=True)
                    if random.random() < 0.4:
                        next_node = random.choice(unvisited)
                    else:
                        next_node = unvisited[0]
                elif start in neighbors:
                    next_node = start
                    found = True
                else:
                    break

                if found:
                    if len(path) >= 3:
                        if best_cycle is None or len(path) > len(best_cycle):
                            best_cycle = path[:]
                    break

                path.append(next_node)
                visited.add(next_node)
                current = next_node

            if found and len(path) >= 3:
                if best_cycle is None or len(path) > len(best_cycle):
                    best_cycle = path[:]

        return best_cycle

    def _find_path_avoiding_edge(u, v, adj, remaining_edges, exclude_edge):
        """Find a path from v to u avoiding the specific edge (u,v)."""
        # BFS from v to u, not using the edge (u,v)
        from collections import deque

        parent = {v: None}
        queue = deque([v])

        while queue:
            current = queue.popleft()
            if current == u:
                # Reconstruct path
                path = []
                node = u
                while node != v:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj[current]:
                if nb not in parent:
                    e = tuple(sorted((current, nb)))
                    if e == exclude_edge:
                        continue
                    if e in remaining_edges:
                        parent[nb] = current
                        queue.append(nb)

        return None

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = find_best_decomposition(edges, n)
        results.append(pieces)
        if time.time() > deadline - 5:
            # Fallback: just use individual edges for remaining graphs
            for spec2 in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results):]:
                n2, p2, seed2 = spec2
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break

    return results
