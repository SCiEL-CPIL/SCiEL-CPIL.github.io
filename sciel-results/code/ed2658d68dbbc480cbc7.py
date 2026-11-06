"""Decompose each instance graph into as few edge-disjoint cycles and single edges as possible."""
import time
import random
from collections import defaultdict, deque

DEADLINE_S = 140.0  # wall-clock budget; keep ~10 s safety margin


def construct():
    t0 = time.time()

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        # Build adjacency
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Edge key helper
        def ekey(a, b):
            return (a, b) if a < b else (b, a)

        pieces = []
        remaining = set(edges)

        # Time check helper
        def time_ok():
            return (time.time() - t0) < DEADLINE_S

        # Greedy: repeatedly find a simple cycle in the remaining graph
        # Use DFS with backtracking (bounded) to find long cycles
        def find_cycle(adj, remaining, max_nodes=30):
            """Find a simple cycle. Returns list of vertices [v0, v1, ..., v_{k-1}] with
            edges (v0,v1),...,(v_{k-2},v_{k-1}),(v_{k-1},v0) all in remaining, or None."""
            # Pick a starting vertex with high degree
            candidates = [v for v in adj if adj[v] and any(ekey(v, w) in remaining for w in adj[v])]
            if not candidates:
                return None
            # Prefer vertices with higher degree in remaining
            def deg_in_remaining(v):
                return sum(1 for w in adj[v] if ekey(v, w) in remaining)
            candidates.sort(key=deg_in_remaining, reverse=True)

            best_cycle = None
            for start in candidates[:6]:  # try a few starts
                if not time_ok():
                    break
                # DFS to find a cycle back to start
                # State: current vertex, path (list), visited set
                path = [start]
                visited = {start}
                # Stack for backtracking: (vertex, neighbor_index)
                # We'll do iterative DFS with backtracking to find ANY simple cycle
                # but prefer longer ones by exploring in degree order
                stack = [(start, list(adj[start]))]
                # We need to ensure edges are in remaining
                # Filter neighbors to only those connected via remaining edges
                def live_neighbors(v):
                    return [w for w in adj[v] if ekey(v, w) in remaining]

                # Reset
                stack = [(start, live_neighbors(start))]
                path = [start]
                visited = {start}

                while stack:
                    if not time_ok():
                        return best_cycle
                    v, neighbors = stack[-1]
                    if not neighbors:
                        stack.pop()
                        if len(path) > 1:
                            path.pop()
                            visited.discard(v)
                        continue
                    w = neighbors.pop()
                    if w == start:
                        if len(path) >= 2:
                            cycle = path[:]
                            if best_cycle is None or len(cycle) > len(best_cycle):
                                best_cycle = cycle
                            return best_cycle
                        # path has only start, can't form cycle with just one edge
                        continue
                    if w in visited:
                        continue
                    # Extend
                    path.append(w)
                    visited.add(w)
                    stack.append((w, live_neighbors(w)))

            return best_cycle

        # Iteratively remove cycles
        max_iter = 1000
        for _ in range(max_iter):
            if not time_ok():
                break
            if not remaining:
                break
            # Rebuild adjacency for remaining
            adj_rem = defaultdict(list)
            for u, v in remaining:
                adj_rem[u].append(v)
                adj_rem[v].append(u)

            cycle = find_cycle(adj_rem, remaining)
            if cycle and len(cycle) >= 3:
                # Verify all edges in cycle are in remaining
                valid = True
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i + 1) % len(cycle)]
                    if ekey(a, b) not in remaining:
                        valid = False
                        break
                if valid:
                    pieces.append(cycle)
                    for i in range(len(cycle)):
                        a = cycle[i]
                        b = cycle[(i + 1) % len(cycle)]
                        remaining.discard(ekey(a, b))
                else:
                    break
            else:
                break

        # Remaining edges as single pieces
        for e in remaining:
            if not time_ok():
                break
            pieces.append([e[0], e[1]])

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if not time_ok():
            # Fallback: just return all edges as singles
            for s in range(len(results), 3):
                sp = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[s]
                results.append([[u, v] for u, v in instance(*sp)])
            return results
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)

    return results
