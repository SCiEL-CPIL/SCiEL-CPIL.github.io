"""Decompose graphs into cycles and single edges to minimize total pieces."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    START_TIME = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_greedy_with_restarts(edges, n, max_restarts=50, seed=42):
        """
        Greedy cycle extraction with randomized restarts.
        Each restart: repeatedly find a cycle in the remaining graph, extract it,
        until no cycles remain. Leftover edges become single-edge pieces.
        We try multiple random orderings / heuristics to get more/better cycles.
        """
        best_pieces = None
        best_count = float('inf')
        rng = random.Random(seed)

        for restart in range(max_restarts):
            if time.time() > DEADLINE - 5.0:
                break

            # Work on a copy of the edge set
            remaining = set(edges)
            # Build adjacency for quick lookup
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)

            pieces = []

            # Repeatedly extract cycles
            while remaining:
                if time.time() > DEADLINE - 5.0:
                    break
                # Find a cycle using randomized DFS/BFS
                cycle = None
                # Try from a random starting vertex
                start_candidates = list(adj.keys())
                rng.shuffle(start_candidates)

                for start in start_candidates:
                    if len(adj[start]) < 2:
                        continue
                    # BFS to find a cycle
                    visited = {start: None}
                    parent = {start: None}
                    queue = deque([start])
                    found = False
                    while queue and not found:
                        u = queue.popleft()
                        for v in list(adj[u]):
                            if v not in visited:
                                visited[v] = u
                                parent[v] = u
                                queue.append(v)
                            elif v != parent.get(u) and v != start:
                                # Found a back edge -> cycle
                                # Reconstruct cycle from u back to v
                                cycle = [u]
                                cur = u
                                while cur != v:
                                    cur = parent.get(cur)
                                    if cur is None:
                                        break
                                    cycle.append(cur)
                                cycle.append(v)
                                # cycle is now v -> ... -> u, and edge (u,v) closes it
                                # Actually cycle list: [u, ..., v] and edge (u,v) exists
                                # The cycle vertices are v, ..., u (distinct)
                                if len(set(cycle)) >= 3:
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        cycle = list(set(cycle))  # ensure distinct
                        # Order the cycle properly: v, ..., u
                        # Let's reconstruct properly
                        cycle_ordered = [v]
                        cur = v
                        while cur != u:
                            cur = parent.get(cur)
                            if cur is None:
                                break
                            cycle_ordered.append(cur)
                        cycle_ordered.append(u)
                        # cycle_ordered: v -> ... -> u, and edge (u,v) closes
                        if len(cycle_ordered) >= 3:
                            cycle = cycle_ordered
                        break
                    if len(cycle) >= 3:
                        break

                if cycle is None or len(cycle) < 3:
                    break

                # Extract the cycle's edges
                cycle_edges = set()
                for i in range(len(cycle)):
                    a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((a, b)))
                    cycle_edges.add(e)

                # Verify all edges are in remaining
                if not cycle_edges.issubset(remaining):
                    # This shouldn't happen, but skip
                    break

                # Remove cycle edges
                remaining -= cycle_edges
                for u, v in cycle_edges:
                    adj[u].discard(v)
                    adj[v].discard(u)

                # Clean up empty adjacency
                for x in list(adj.keys()):
                    if not adj[x]:
                        del adj[x]

                pieces.append(cycle)

            # Leftover edges as single-edge pieces
            for e in remaining:
                pieces.append(list(e))

            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces

        if best_pieces is None:
            # Fallback: all edges as single pieces
            best_pieces = [list(e) for e in edges]

        return best_pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    for i, spec in enumerate(specs):
        if time.time() > DEADLINE - 5.0:
            # Not enough time; use trivial decomposition for remaining
            for j in range(i, len(specs)):
                edges = instance(*specs[j])
                results.append([list(e) for e in edges])
            break

        edges = instance(*spec)
        n = spec[0]
        pieces = decompose_greedy_with_restarts(edges, n, max_restarts=100, seed=42 + i)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
